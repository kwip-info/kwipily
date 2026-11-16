import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTemplateFieldMatchingTests:XCTestCase {
    private let names=["SourceRowID","InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"]
    private var source:WorkflowNodeAddress {.init(domain:.data,localID:"source")}
    private func owner(_ document:WorkflowDocument,root:URL)throws->WorkflowAuthoringStore {
        let library=ProjectLibrary(directory:root)
        _ = try library.saveSource(.workflow(document),id:document.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try .init(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testExplicitNameMappingIsOneUndoablePortableEditAndPreservesAllOtherNodes()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let original=try WorkflowTemplates.instantiate(id:"sales-exceptions"),owner=try owner(original,root:root)
        let observed=names.map{$0=="CustomerID" ? "customer_ref":$0}
        let plan=try WorkflowTemplateFieldMatchPlan(document:original,address:source,observed:observed)
        XCTAssertEqual(plan.initial["CustomerID"],"")
        var matches=plan.initial;matches["CustomerID"]="customer_ref"
        let commands=try plan.commands(in:original,matches:matches)
        XCTAssertNotNil(owner.applyBatch(commands,lease:owner.lease()))
        let mapped=owner.document
        XCTAssertEqual(mapped.revision,original.revision+1)
        XCTAssertEqual(mapped.nodes.count,original.nodes.count+1)
        XCTAssertEqual(mapped.edges.count,original.edges.count+1)
        XCTAssertEqual(mapped.nodes.filter{$0.address != source && $0.address != plan.mapperAddress},original.nodes.filter{$0.address != source})
        XCTAssertEqual(Set(mapped.edges.filter{$0.from.address==source}.map(\.to)),[.init(address:plan.mapperAddress,port:"table")])
        let mapper=try XCTUnwrap(mapped.nodes.first{$0.address==plan.mapperAddress})
        guard case .dataOperator("kwip.data.columns",let config)=mapper.payload,case .array(let rows)=config["rows"] else {return XCTFail("Named Columns adapter required")}
        XCTAssertTrue(rows.contains(.object(["source":.string("customer_ref"),"outputName":.string("CustomerID"),"included":.bool(true),"targetType":.string("keep")])))
        guard case .dataOperator(_,let input)=mapped.nodes.first(where:{$0.address==source})!.payload,case .object(let types)=input["schemaOverrides"] else{return XCTFail("Missing source types")}
        XCTAssertNil(types["CustomerID"]);XCTAssertEqual(types["customer_ref"],.string("String"))
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(mapped)),mapped)
        let reloaded=try self.owner(mapped,root:root.appendingPathComponent("reload"))
        XCTAssertEqual(reloaded.document,mapped)
        let reuse=try WorkflowTemplateFieldMatchPlan(document:mapped,address:source,observed:observed.reversed())
        XCTAssertEqual(reuse.initial["CustomerID"],"customer_ref")
        XCTAssertTrue(try reuse.commands(in:mapped,matches:reuse.initial).isEmpty,"Header reordering must not require edits or reinterpret values by position")
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,original.nodes);XCTAssertEqual(owner.document.edges,original.edges)
    }
    func testExactMatchesAreNoOpAndInvalidOrInventedMappingsRefuse()throws {
        let document=try WorkflowTemplates.instantiate(id:"sales-exceptions")
        let plan=try WorkflowTemplateFieldMatchPlan(document:document,address:source,observed:names)
        XCTAssertTrue(try plan.commands(in:document,matches:plan.initial).isEmpty)
        var duplicate=plan.initial;duplicate["CustomerID"]="Country"
        XCTAssertThrowsError(try plan.commands(in:document,matches:duplicate))
        var absent=plan.initial;absent["CustomerID"]="invented"
        XCTAssertThrowsError(try plan.commands(in:document,matches:absent))
        XCTAssertThrowsError(try WorkflowTemplateFieldMatchPlan(document:document,address:source,observed:names+["Country"]))
        var ordinary=document;ordinary.templateOrigin=nil
        XCTAssertThrowsError(try WorkflowTemplateFieldMatchPlan(document:ordinary,address:source,observed:names))
    }
    func testTemplatesWithoutReaderOverridesUseTheirActualFirstColumnsContract()throws {
        let doc=try WorkflowTemplates.instantiate(id:"customer-segmentation")
        let plan=try WorkflowTemplateFieldMatchPlan(document:doc,address:source,observed:names)
        XCTAssertEqual(Set(plan.required),Set(names.filter{$0 != "SourceRowID"}))
        XCTAssertTrue(try plan.commands(in:doc,matches:plan.initial).isEmpty)
        var edited=doc
        let index=try XCTUnwrap(edited.nodes.firstIndex{$0.address==source})
        guard case .dataOperator(let id,var config)=edited.nodes[index].payload else{return XCTFail()}
        config["columns"] = .array([.string("Country")]);edited.nodes[index].payload = .dataOperator(operatorID:id,config:config)
        XCTAssertThrowsError(try WorkflowTemplateFieldMatchPlan(document:edited,address:source,observed:names))
    }
    func testOwnedDraftBlocksRunAndRevertClearsOnlyItsToken()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let document=try WorkflowTemplates.instantiate(id:"sales-exceptions"),owner=try owner(document,root:root)
        let sources=FieldMatchSourceProbe(owner:owner,columns:names.map{$0=="CustomerID" ? "customer_ref":$0})
        let step=try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first{$0.address==source})
        let session=try WorkflowTemplateFieldMatchSession(step:step,owner:owner,sources:sources)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertFalse(session.canApply)
        session.choose("CustomerID","customer_ref")
        XCTAssertTrue(session.canApply)
        let other=UUID()
        XCTAssertTrue(owner.setInspectorEditPending(contextID:other,address:source,key:"other",pending:true,lease:owner.lease()))
        XCTAssertFalse(session.apply())
        XCTAssertEqual(owner.document,document)
        session.discard();XCTAssertTrue(owner.hasPendingInspectorEdits,"Revert must not clear another inspector's draft")
        owner.clearDataEdits(contextID:other);XCTAssertFalse(owner.hasPendingInspectorEdits)
    }
    func testHeaderAndSourceIdentityRevalidatedBeforeApplyingWithoutOpeningOrRunningAnything()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let document=try WorkflowTemplates.instantiate(id:"sales-exceptions"),owner=try owner(document,root:root)
        let sources=FieldMatchSourceProbe(owner:owner,columns:names.map{$0=="CustomerID" ? "customer_ref":$0})
        let step=try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first{$0.address==source})
        let session=try WorkflowTemplateFieldMatchSession(step:step,owner:owner,sources:sources)
        session.choose("CustomerID","customer_ref")
        sources.preview = .init(columns:session.plan.observed,rows:[],rowCount:0,sourceSHA256:String(repeating:"b",count:64))
        XCTAssertFalse(session.canApply);XCTAssertFalse(session.apply());XCTAssertEqual(owner.document,document)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        session.discard();XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(sources.mutations,0)
    }
    func testActualSessionAppliesAsOneBatchAndBusyOrForeignSourceRefuses()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let document=try WorkflowTemplates.instantiate(id:"sales-exceptions"),owner=try owner(document,root:root)
        let columns=names.map{$0=="CustomerID" ? "customer_ref":$0}
        let sources=FieldMatchSourceProbe(owner:owner,columns:columns)
        let step=try XCTUnwrap(WorkflowTemplateSetupPlan.steps(in:document).first{$0.address==source})
        let foreign=FieldMatchSourceProbe(owner:owner,columns:columns,sessionID:UUID())
        XCTAssertThrowsError(try WorkflowTemplateFieldMatchSession(step:step,owner:owner,sources:foreign))
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        let session=try WorkflowTemplateFieldMatchSession(step:step,owner:owner,sources:sources)
        session.choose("CustomerID","customer_ref")
        owner.setExecutionBusy(true);XCTAssertFalse(session.apply());XCTAssertEqual(owner.document,document)
        owner.setExecutionBusy(false);XCTAssertTrue(session.apply())
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(owner.document.revision,document.revision+1)
        XCTAssertEqual(sources.mutations,0,"Field matching must not acquire grants, run a job or implicitly inspect files")
    }
    func testEditedMappingAndChangedSourceCannotBeOverwrittenByOldPlan()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let document=try WorkflowTemplates.instantiate(id:"sales-exceptions"),owner=try owner(document,root:root)
        let observed=names.map{$0=="CustomerID" ? "customer_ref":$0}
        let plan=try WorkflowTemplateFieldMatchPlan(document:document,address:source,observed:observed)
        var matches=plan.initial;matches["CustomerID"]="customer_ref"
        XCTAssertNotNil(owner.applyBatch(try plan.commands(in:document,matches:matches),lease:owner.lease()))
        var edited=owner.document
        let index=try XCTUnwrap(edited.nodes.firstIndex{$0.address==plan.mapperAddress})
        guard case .dataOperator(let id,var config)=edited.nodes[index].payload else{return XCTFail()}
        config["trimText"] = .bool(true);edited.nodes[index].payload = .dataOperator(operatorID:id,config:config)
        XCTAssertThrowsError(try WorkflowTemplateFieldMatchPlan(document:edited,address:source,observed:observed))
        XCTAssertThrowsError(try plan.commands(in:owner.document,matches:matches))
    }

}

@MainActor private final class FieldMatchSourceProbe:WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage=""
    var mutations=0
    var preview:SourceInspectionResult?
    var changes:AnyPublisher<Void,Never>{Empty(completeImmediately:false).eraseToAnyPublisher()}
    init(owner:WorkflowAuthoringStore,columns:[String],sessionID:UUID?=nil){
        projectID=owner.projectID;self.sessionID=sessionID ?? owner.sessionID
        preview = .init(columns:columns,rows:[],rowCount:0,sourceSHA256:String(repeating:"a",count:64))
    }
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema]{[]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String?{"synthetic.csv"}
    func sourceInspectionPreview(for address:WorkflowNodeAddress)->SourceInspectionResult?{preview}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool{mutations+=1;return false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool{mutations+=1;return false}
}
