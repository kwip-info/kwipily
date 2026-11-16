import Foundation
import Combine
import KwipMLCore

/// Per-view page state. The owner supplies an already verified artifact reference
/// and a selection predicate; neither a canvas nor original source bindings are read.
@MainActor final class TableViewQuerySession: ObservableObject {
    struct Artifact: Equatable {
        let indexURL: URL
        let indexSHA256: String
        let sourceSHA256: String
        var inputKind: String? = nil
        var tokenVocabulary:TokenVocabularyResolver? = nil
        var retainedSegments: [WorkflowArchiveFile] = []
        var workloadPolicy: ExecutionWorkloadPolicy? = nil
    }
    @Published private(set) var page: TableViewPage?
    @Published private(set) var navigation = TableViewNavigationState()
    @Published private(set) var isBusy = false
    @Published private(set) var statusMessage = ""
    let imageResolver: FileImageArtifactResolver?
    var tokenVocabulary:TokenVocabularyResolver? {artifact.tokenVocabulary}
    private var artifact: Artifact
    private var formula:[String:JSONValue]?
    func formulaSample(configuration:[String:JSONValue])->TableViewQuerySession {
        formula=configuration;return self
    }
    private let workspace: URL
    private let accountingRoot: URL?
    private let projectID: String
    private let runtime: RuntimeLocation
    private let storageLimitBytes: Int64?
    private let reservation: WorkflowExecutionReservation
    private let accepts: () -> Bool
    private var token: UUID?
    private var task: Task<Void, Never>?
    private var worker: WorkflowWorkerHandle?
    private var cancelled = false
    private var generation = UUID()
    private var retained: TableViewQuerySession?
    private var cellRead:TableViewQuerySession?
    private var completion:((TableViewPage?,String)->Void)?
    private var deferredQuery:(offset:Int,search:String,sort:TableGridSort)?
    /// Reuse the artifact owner predicate for automation polling as well as worker completion.
    var isCurrent:Bool {accepts()}
    var waitingForExecution:Bool {deferredQuery != nil}
    /// A visible viewer retries its exact captured query once the run releases
    /// the shared reservation. Never poll or reopen the original source.
    @discardableResult func resumeAfterExecution()->Bool {
        guard !isBusy else {return false}
        if let deferredQuery {return load(offset:deferredQuery.offset,search:deferredQuery.search,sort:deferredQuery.sort)}
        guard page==nil,statusMessage.isEmpty else {return false}
        return load(offset:navigation.offset,search:navigation.search,sort:navigation.sort)
    }

    init(artifact: Artifact, workspace: URL, accountingRoot: URL? = nil, projectID: String,
         runtime: RuntimeLocation, storageLimitBytes: Int64? = nil,
         reservation: WorkflowExecutionReservation, imageResolver: FileImageArtifactResolver? = nil, accepts: @escaping () -> Bool) {
        self.artifact = artifact; self.workspace = workspace; self.accountingRoot = accountingRoot
        self.projectID = projectID; self.runtime = runtime; self.storageLimitBytes = storageLimitBytes
        self.reservation = reservation; self.accepts = accepts; self.imageResolver = imageResolver
    }
    @discardableResult func load(offset: Int = 0, search: String = "", sort: TableGridSort = .init(), fullCellColumn:Int? = nil, fullCellStart:Int=0) -> Bool {
        guard !isBusy else { statusMessage = "Wait for the current table query or cancel it first."; return false }
        deferredQuery=nil
        guard accepts() else { statusMessage = "This table is no longer current. Select its recorded result again."; return false }
        var query = TableViewQuery(offset: offset, limit: 100, search: search,
                                   sourceSHA256: artifact.sourceSHA256, indexSHA256: artifact.indexSHA256, sortColumn: sort.column, sortDescending: sort.descending)
        query.inputKind = artifact.inputKind
        if artifact.inputKind != nil { query.schemaVersion = 3 }
        if let formula {query.schemaVersion=4;query.formula=formula;query.limit=8;query.offset=0;query.search="";query.sortColumn=nil;query.sortDescending=false}
        if let fullCellColumn {query.schemaVersion=5;query.limit=1;query.fullCellColumn=fullCellColumn;query.fullCellStart=fullCellStart}
        let window: ExecutionBudgetWindow
        do {
            try query.validate(workload: artifact.workloadPolicy)
            window = try ExecutionBudgetWindow(policy: ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 30, resources: try artifact.workloadPolicy.map { try ExecutionResourcePolicy(cpuThreads: 1, workloadPolicy: $0) }), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        } catch { statusMessage = error.localizedDescription; return false }
        guard let acquired = reservation.acquire() else {
            deferredQuery=(offset,search,sort)
            statusMessage = "Waiting for the active operation. This captured table will load when it finishes."
            return false
        }
        navigation.begin(offset: offset, search: search, sort: sort, pageOffset: page?.offset, nextOffset: page?.nextOffset)
        token = acquired; isBusy = true; cancelled = false; retained = self; page = nil
        let requestID = UUID(); generation = requestID
        statusMessage = "Loading table…"
        task = Task { [self] in
            defer { if generation == requestID { task = nil } }
            do {
                let handle = try await TableViewQueryExecution.launch(workspace: workspace, accountingRoot: accountingRoot,
                    projectID: projectID, runtime: runtime, window: window, indexURL: artifact.indexURL, query: query, retainedSegments: artifact.retainedSegments,
                    storageLimitBytes: storageLimitBytes, onMaterializedIndex: { [weak self] index in
                        guard let self, self.generation == requestID, !self.cancelled, self.accepts() else { return }
                        var updated=index;updated.tokenVocabulary=self.artifact.tokenVocabulary;updated.workloadPolicy=self.artifact.workloadPolicy;self.artifact = updated
                    }, onExit: { [self] code, message, result in
                        finish(requestID, code: code, message: message, result: result)
                    })
                if isBusy, generation == requestID { worker = handle; if cancelled { handle.cancel() } }
            } catch { finish(requestID, code: -1, message: error.localizedDescription, result: nil) }
        }
        return true
    }
    func cancel() {
        if deferredQuery != nil {deferredQuery=nil;statusMessage="Table query cancelled."}
        guard isBusy else { return }
        cancelled = true; task?.cancel(); worker?.cancel()
        statusMessage = "Cancelling table query…"
    }
    func readFullCell(row:Int,column:Int,start:Int=0,completion:@escaping(CapturedCellPage)->Void) {
        cancelCellRead()
        let child=TableViewQuerySession(artifact:artifact,workspace:workspace,accountingRoot:accountingRoot,projectID:projectID,runtime:runtime,storageLimitBytes:storageLimitBytes,reservation:reservation,accepts:accepts)
        cellRead=child
        child.completion={ [weak self,weak child] page,message in
            guard let self,let child,self.cellRead === child else {return}
            self.cellRead=nil
            if let value=page?.rows.first?.first,let columns=page?.columns.first {
                completion(.init(text:CapturedDataGrid.cell(WorkflowCapturedDataOutput.render(value),column:columns.mapValues(WorkflowCapturedDataOutput.render)).text,start:page?.cellOffset ?? 0,total:page?.cellTotalCharacters ?? 0))
            } else {completion(.unavailable(message))}
        }
        if !child.load(offset:row,fullCellColumn:column,fullCellStart:start) {cellRead=nil;completion(.unavailable(child.statusMessage))}
    }
    func cancelCellRead(){cellRead?.cancel();cellRead=nil}
    func cancelIfUnowned() {if !accepts() {cancel()}}
    func invalidate() { cancelCellRead();cancel(); page = nil }
    private func finish(_ requestID: UUID, code: Int32, message: String, result: TableViewPage?) {
        guard isBusy, generation == requestID else { return }
        defer {
            if let token { reservation.release(token) }
            token = nil; worker = nil; isBusy = false; retained = nil
            completion?(page,statusMessage);completion=nil
        }
        guard !cancelled, accepts() else { statusMessage = "Table query cancelled."; return }
        guard code == 0, let result else {
            statusMessage = message.isEmpty ? "The table page could not be verified." : message
            return
        }
        page = result; statusMessage = ""
    }
}
