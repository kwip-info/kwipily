import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTemplateSetupTests: XCTestCase {
    func testItemSalesSetupOnlyAsksForSourceAndReportFolder() throws {
        let document=try WorkflowTemplates.instantiate(id:"item-sales-reconciliation")
        let steps=WorkflowTemplateSetupPlan.steps(in:document)
        XCTAssertEqual(steps.filter {$0.action == .chooseSource}.count,1)
        XCTAssertEqual(steps.filter {$0.action == .chooseOutput}.count,3)
        XCTAssertFalse(steps.contains {$0.action == .configure},"The graph derives its two months; do not add a redundant date wizard")
        let source=try XCTUnwrap(steps.first {$0.action == .chooseSource})
        XCTAssertEqual(WorkflowTemplateSetupPlan.missingFields(for:source,in:document,observedColumns:["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"]),["SourceRowID"])
    }
    func testChronologicalTrainingDatesAreExplicitSetupInsteadOfHiddenDemoDefaults() throws {
        let document=try WorkflowTemplates.instantiate(id:"rental-demand")
        let step=try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first {$0.address.localID=="split"})
        XCTAssertEqual(step.action,.configure)
        XCTAssertEqual(step.title,"Training and validation dates")
        XCTAssertEqual(WorkflowTemplateSetupPlan.configurationSummary(step,in:document),"Train through 2011-01-31T23:59:59.999Z · validate through 2011-02-28T23:59:59.999Z. Review these dates for your data.")
        let random=try WorkflowTemplates.instantiate(id:"complaint-routing")
        XCTAssertFalse(WorkflowTemplateSetupPlan.steps(in:random).contains {$0.address.localID=="split"},"A random split has no date question; don't add universal wizard steps.")
    }
    func testConnectionAndModelSetupNeverBecomeLocalFileChoosers() throws {
        let weather = try WorkflowTemplates.instantiate(id: "weather-briefing")
        let steps = WorkflowTemplateSetupPlan.steps(in: weather)
        XCTAssertEqual(steps.first?.action, .configure)
        XCTAssertEqual(steps.first?.address.localID, "forecast")
        XCTAssertEqual(steps.last?.action, .chooseOutput)
        let ocr = try WorkflowTemplates.instantiate(id: "document-ocr")
        XCTAssertTrue(WorkflowTemplateSetupPlan.steps(in: ocr).contains { $0.action == .configure })
        XCTAssertEqual(WorkflowTemplateSetupPlan.steps(in: ocr).filter { $0.action == .chooseSource }.count, 1)
        var ordinary = weather; ordinary.templateOrigin = nil
        XCTAssertTrue(WorkflowTemplateSetupPlan.steps(in: ordinary).isEmpty)
    }

    func testEditedExternalOutputUsesConnectionConfigurationInsteadOfFolderGrant() throws {
        var document = try WorkflowTemplates.instantiate(id: "sales-exceptions")
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address.localID == "cancelout" })
        guard case .dataOperator(let id, var config) = document.nodes[index].payload else { return XCTFail("Expected table output") }
        config["destination"] = .string("api")
        document.nodes[index].payload = .dataOperator(operatorID:id,config:config)
        let step = try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first { $0.address.localID == "cancelout" })
        XCTAssertEqual(step.action,.configure)
        XCTAssertEqual(WorkflowTemplateSetupPlan.steps(in:document).filter { $0.action == .chooseOutput }.count,2)
    }

    func testReadableSourceDoesNotHideMissingRequiredTemplateFields() throws {
        let document = try WorkflowTemplates.instantiate(id:"sales-exceptions")
        let source = try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first { $0.action == .chooseSource })
        let names = ["SourceRowID","InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","Country"]
        XCTAssertEqual(WorkflowTemplateSetupPlan.missingFields(for:source,in:document,observedColumns:names),["CustomerID"])
        XCTAssertEqual(WorkflowTemplateSetupPlan.missingFields(for:source,in:document,observedColumns:names+["CustomerID"]),[])
        XCTAssertNil(WorkflowTemplateSetupPlan.missingFields(for:source,in:document,observedColumns:nil),"No cached inspection is not a successful field check.")
        XCTAssertEqual(WorkflowTemplateSetupPlan.missingFields(for:source,in:document,observedColumns:names+["customerid"]),["CustomerID"],"Do not guess that a different name has the same business meaning.")
        let output = try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first { $0.action == .chooseOutput })
        XCTAssertNil(WorkflowTemplateSetupPlan.missingFields(for:output,in:document,observedColumns:[]))
    }

    func testSetupUsesCurrentOwnerAndDoesNotAcquireAccessDuringPlanning() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = ProjectLibrary(directory: root)
        let document = try WorkflowTemplates.instantiate(id: "sales-exceptions")
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID,
            modelRegistry: .builtIn(), dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
        let sources = TemplateSetupSourceProbe(owner: owner)
        let step = try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in: owner.document).first)
        XCTAssertTrue(sources.actions.isEmpty)
        let lease = owner.lease()
        XCTAssertTrue(WorkflowTemplateSetupPlan.focus(step,owner:owner,sources:sources,lease:lease))
        XCTAssertTrue(sources.actions.isEmpty,"Reviewing fields must not open a chooser or acquire access.")
        XCTAssertTrue(WorkflowTemplateSetupPlan.perform(step, owner: owner, sources: sources, lease: lease))
        XCTAssertEqual(sources.actions, [step.address])
        XCTAssertEqual(owner.selection.nodeAddresses, [step.address])
        XCTAssertEqual(owner.document, document, "Setup navigation must not rewrite the workflow or run it.")
        owner.setExecutionBusy(true)
        XCTAssertFalse(WorkflowTemplateSetupPlan.perform(step, owner: owner, sources: sources, lease: lease))
        owner.setExecutionBusy(false)
        let foreign = TemplateSetupSourceProbe(owner: owner, sessionID: UUID())
        XCTAssertFalse(WorkflowTemplateSetupPlan.perform(step, owner: owner, sources: foreign, lease: lease))
        var wrong = step; wrong.action = .chooseOutput
        XCTAssertFalse(WorkflowTemplateSetupPlan.perform(wrong, owner: owner, sources: sources, lease: lease))
        XCTAssertEqual(sources.actions.count, 1)
    }
}

@MainActor private final class TemplateSetupSourceProbe: WorkflowDataSources {
    let projectID: String
    let sessionID: UUID
    var isBusy = false
    var statusMessage = ""
    var actions: [WorkflowNodeAddress] = []
    var changes: AnyPublisher<Void, Never> { Empty(completeImmediately: false).eraseToAnyPublisher() }
    init(owner: WorkflowAuthoringStore, sessionID: UUID? = nil) {
        projectID = owner.projectID; self.sessionID = sessionID ?? owner.sessionID
    }
    func recordedSchema(for address: WorkflowNodeAddress, in document: WorkflowDocument) -> [DataColumnSchema] { [] }
    func bindingDisplayName(for address: WorkflowNodeAddress) -> String? { nil }
    func chooseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool { actions.append(address); return true }
    func inspectSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool { false }
}
