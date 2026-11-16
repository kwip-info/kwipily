import SwiftUI
import KwipMLCore

/// A bounded page, not the full materialized table, is held by this view.
struct TableViewOutputView: View {
    @ObservedObject var session: TableViewQuerySession
    var executionBusy=false
    var imageResolver: FileImageArtifactResolver? = nil
    @State private var mediaSelection:MediaArtifactSelection?
    @State private var imageSelection: FileImageSelection?
    @State private var sequenceSelection:SequenceWindowSelection?
    @State private var generatedTokens:[Int64]?
    @State private var generatedVocabulary:[String]?
    @State private var sequenceFailure:String?
    private var resolvedImages: FileImageArtifactResolver? { imageResolver ?? session.imageResolver }
    @State private var search = ""
    @State private var selectedRow: Int?
    private var sort: TableGridSort { session.navigation.sort }
    @FocusState private var searchFocused: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let generatedTokens {
                Button("Back to Output") {self.generatedTokens=nil;self.generatedVocabulary=nil}.buttonStyle(FlatButtonStyle())
                TokenSequenceViewer(ids:generatedTokens,vocabulary:generatedVocabulary)
            } else if let sequenceSelection {
                SequenceWindowViewer(selection:sequenceSelection) {self.sequenceSelection=nil}
            } else {
            ViewThatFits(in:.horizontal) {
                HStack(spacing:10) { tableSummary.fixedSize(); Spacer(minLength:8); tableNavigation.fixedSize() }
                VStack(alignment:.leading,spacing:8) { tableSummary; tableNavigation }
            }
            if let page = session.page {
                CapturedNativeTable(columns: Self.columns(page), rows: rows(page), selectedRow: $selectedRow, rowHeaderTitle: "Row", sort: sort, onSort: changeSort, tokenVocabulary:session.tokenVocabulary, imageResolver: resolvedImages, onOpenImage: { imageSelection = .init(reference: $0) },onOpenMedia:{mediaSelection = .init(reference:$0)},onOpenMediaRow:{row,ref in
                    mediaSelection = .init(reference:ref,interval:SpeechSegmentPresentation.interval(row:row,columns:Self.columns(page)))
                },onOpenSequence:{ row,column in openSequence(row,column:column,page:page) },onInspectCell:{row,column,start,complete in
                    guard page.cellTruncated,row.cells[column].text.unicodeScalars.count==2000,page.columns[column]["dtype"] == .string("string") else {complete(.slice(row.cells[column].text,start:start));return}
                    complete(.loading)
                    session.readFullCell(row:row.index,column:column,start:start,completion:complete)
                },onCloseCell:{session.cancelCellRead()})
                    .id(ObjectIdentifier(session))
                    .frame(minHeight: 100, maxHeight: .infinity)
                if page.cellTruncated { Text("Long cell values are shortened for display.").font(.caption2).foregroundStyle(MLTheme.muted) }
            } else if session.isBusy {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if session.waitingForExecution {
                Text(session.statusMessage).font(.caption).foregroundStyle(MLTheme.muted)
            } else if executionBusy && session.statusMessage.isEmpty {
                Text("This captured table will load after the active operation finishes.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
                Text("No table page is available.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            if let sequenceFailure {Text(sequenceFailure).font(.caption).foregroundStyle(MLTheme.rose)}
            if !session.statusMessage.isEmpty && !session.waitingForExecution {
                Text(session.statusMessage).font(.caption).foregroundStyle(MLTheme.muted)
            }
            }
        }
        .sheet(item:$mediaSelection) { selected in if let resolvedImages {MediaArtifactPreviewView(reference:selected.reference,resolver:resolvedImages,selectionInterval:selected.interval)} }
        .sheet(item: $imageSelection) { selected in
            if let resolvedImages { FileImagePreviewView(reference: selected.reference, resolver: resolvedImages, ocrRegions:OCRRegionPresentation.regions(page:session.page,image:selected.reference)) }
        }
        .onAppear { restoreSessionPresentation() }
        .onChange(of: ObjectIdentifier(session)) { _, _ in restoreSessionPresentation() }
        .onChange(of: executionBusy) { _,busy in if !busy {_=session.resumeAfterExecution()} }
        .onDisappear {session.cancelCellRead(); session.cancelIfUnowned() }
    }
    private var tableSummary:some View {
        HStack(spacing:10) {
            Text("Table").font(.system(size:12,weight:.semibold)).fixedSize()
            if let page=session.page {
                Text("\(page.matchedRows) of \(page.totalRows) rows · \(page.columns.count) columns")
                    .font(.caption).foregroundStyle(MLTheme.muted).lineLimit(1)
            }
        }
    }
    private var tableNavigation:some View {
        HStack(spacing:8) {
            TextField("Find in table",text:$search).flatField().frame(minWidth:90,idealWidth:190,maxWidth:190)
                .focused($searchFocused).disabled(session.isBusy).onSubmit {applySearch()}
            Button("Find",action:applySearch).buttonStyle(FlatButtonStyle()).disabled(session.isBusy).fixedSize()
            if session.isBusy || session.waitingForExecution {Button("Cancel"){session.cancel()}.buttonStyle(FlatButtonStyle()).fixedSize()}
            Button {previousPage()} label:{Image(systemName:"chevron.left")}
                .buttonStyle(FlatButtonStyle()).disabled(session.isBusy || session.navigation.previousOffsets.isEmpty).accessibilityLabel("Previous table page")
            Button {nextPage()} label:{Image(systemName:"chevron.right")}
                .buttonStyle(FlatButtonStyle()).disabled(session.isBusy || session.page?.nextOffset == nil).accessibilityLabel("Next table page")
        }
    }
    private func restoreSessionPresentation() {
        search = session.navigation.search; selectedRow = nil;sequenceSelection=nil;generatedTokens=nil;generatedVocabulary=nil;sequenceFailure=nil
        if session.page == nil, !session.isBusy, !executionBusy {
            _ = session.load(offset: session.navigation.offset, search: session.navigation.search, sort: sort)
        }
    }
    private func openSequence(_ row:CapturedDataGrid.Row,column:String,page:TableViewPage)->Bool {
        if let resolver=session.tokenVocabulary,let vocabulary=resolver.generationVocabulary?[column],let index=page.columns.firstIndex(where:{$0["name"]?.stringValue==column}),row.cells.indices.contains(index),let ids=row.cells[index].sequence {
            guard ids.allSatisfy({$0>=0 && $0<Int64(vocabulary.count)}) else {sequenceFailure="Captured generation token is outside its vocabulary.";return true}
            generatedTokens=ids;generatedVocabulary=vocabulary;return true
        }
        guard let resolver=session.tokenVocabulary,let config=resolver.windowConfiguration(),[config["inputColumn"]?.stringValue,config["targetColumn"]?.stringValue,config["maskColumn"]?.stringValue].contains(column) else {return false}
        do {sequenceSelection=try .init(row:row,columns:Self.columns(page),configuration:config,vocabulary:resolver.resolve(column:config["inputColumn"]!.stringValue!));sequenceFailure=nil}
        catch {sequenceFailure="Captured sequence pair cannot be verified: "+error.localizedDescription}
        return true
    }
    private func applySearch() {
        searchFocused = false
        if session.load(offset: 0, search: search, sort: sort) { selectedRow = nil }
    }
    private func changeSort(_ column: Int?) {
        if let column, [JSONValue.string("image"),.string("int64Sequence")].contains(session.page?.columns[column]["dtype"] ?? .null) { return }
        let next = sort.toggled(column)
        if session.load(offset: 0, search: session.navigation.search, sort: next) {
            selectedRow = nil
        }
    }
    private func previousPage() {
        guard let offset = session.navigation.previousOffsets.last else { return }
        if session.load(offset: offset, search: session.navigation.search, sort: sort) { selectedRow = nil }
    }
    private func nextPage() {
        guard let page = session.page, let offset = page.nextOffset else { return }
        if session.load(offset: offset, search: session.navigation.search, sort: sort) { selectedRow = nil }
    }
    static func columns(_ page: TableViewPage) -> [CapturedDataGrid.Column] {
        page.columns.enumerated().map { index, column in
            let name: String, dtype: String, nullable: String
            if case .string(let value) = column["name"] { name = value } else { name = "Column \(index + 1)" }
            if case .string(let value) = column["dtype"] { dtype = value } else { dtype = "unknown" }
            if case .bool(let value) = column["nullable"] { nullable = value ? "nullable" : "required" } else { nullable = "unknown" }
            return .init(id: index, name: name, label: column["displayLabel"]?.stringValue ?? name, dtype: dtype, nullability: nullable)
        }
    }
    private func rows(_ page: TableViewPage) -> [CapturedDataGrid.Row] {
        zip(page.positions, page.rows).map { position, values in
            .init(index: position, cells: values.enumerated().map { index, value in
                CapturedDataGrid.cell(WorkflowCapturedDataOutput.render(value), column: page.columns[index].mapValues(WorkflowCapturedDataOutput.render))
            })
        }
    }
}
