import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationDeclarationTests:XCTestCase {
    private let frozen=WorkflowNodeAddress(domain:.control,localID:"frozen")
    private let terminal=WorkflowNodeAddress(domain:.control,localID:"evaluate")
    private func address(_ id:String)->WorkflowNodeAddress {.init(domain:.data,localID:id)}
    private func registries()throws->(ComponentRegistry,DataFlowOperatorRegistry) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try (.builtIn(),.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func fixture()throws->(WorkflowDocument,DataFlowDocument) {
        var document=try WorkflowExamples.preparedGroupSplitClassifier(device:"cpu")
        let (model,data)=try registries()
        let archive=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:4).dataflow
        document.schemaVersion=5
        document.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:terminal,payload:.evaluateTest(config:[:]))]
        document.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:terminal,port:"model"),kind:.frozenModel),.init(from:.init(address:address("split"),port:"test"),to:.init(address:terminal,port:"test"),kind:.table)]
        return (document,archive)
    }
    private func validate(_ document:WorkflowDocument,_ archive:DataFlowDocument)throws {
        let (model,data)=try registries()
        let plan=try WorkflowEvaluationCompiler.compile(document,evaluate:terminal,modelRegistry:model,dataRegistry:data)
        try WorkflowEvaluationDeclaration.validate(document:document,plan:plan,archivedDataflow:archive)
    }
    func testPresentationEditsAndUnrelatedTrainingEditsDoNotChangeArchivedSelection()throws {
        var (document,archive)=try fixture()
        try validate(document,archive)
        for index in document.nodes.indices {document.nodes[index].name="Renamed \(index)";document.nodes[index].x+=17;document.nodes[index].y-=9}
        for index in document.edges.indices {document.edges[index].id=UUID().uuidString}
        let setup=try XCTUnwrap(document.nodes.firstIndex {if case .runSetup=$0.payload{return true};return false})
        document.nodes[setup].payload = .runSetup(config:["epochs":.integer(9)])
        try validate(document,archive)
    }
    func testChangedSourceSplitAndPrepartitionOperatorReject()throws {
        for id in ["observations","split","cast"] {
            var (document,archive)=try fixture()
            let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==address(id)})
            guard case .dataOperator(let op,var config)=document.nodes[index].payload else {return XCTFail("Missing fixture operator")}
            switch id {
            case "observations":config["separator"] = .string(";")
            case "split":config["seed"] = .integer(999)
            default:
                guard case .array(var rows)=config["rows"],case .object(var row)=rows.first else {return XCTFail("Current Columns rules")}
                row["targetType"] = .string("int64");rows[0] = .object(row);config["rows"] = .array(rows)
            }
            document.nodes[index].payload = .dataOperator(operatorID:op,config:config)
            XCTAssertThrowsError(try validate(document,archive))
        }
    }
    func testPrunedCurrentBranchCannotSilentlySelectRawTestInstead()throws {
        var (document,archive)=try fixture()
        let inspected=address("test-inspection")
        let identity:[String:JSONValue]=["rows":.array([]),"castFailure":.string("error"),"trimText":false,"newColumns":.string("keep"),"missingPolicy":.string("preserve")]
        document.nodes.append(.init(address:inspected,payload:.dataOperator(operatorID:"kwip.data.columns",config:identity)))
        let fit=try XCTUnwrap(document.nodes.first {if case .dataOperator("kwip.data.preprocess",_)=$0.payload{return true};return false})
        document.edges.removeAll {$0.to.address==terminal && $0.to.port=="test"}
        document.edges += [.init(from:.init(address:fit.address,port:"test"),to:.init(address:inspected,port:"table"),kind:.table),.init(from:.init(address:inspected,port:"table"),to:.init(address:terminal,port:"test"),kind:.table)]
        XCTAssertThrowsError(try validate(document,archive),"Absent selected branch cannot fall back to the raw split")
        archive.nodes.append(.init(id:"test-inspection",operatorID:"kwip.data.columns",config:identity))
        archive.edges.append(.init(from:.init(node:fit.address.localID,port:"test"),to:.init(node:"test-inspection",port:"table")))
        try validate(document,archive)
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==fit.address})
        guard case .dataOperator(let op,var config)=fit.payload,case .array(var rows)=config["columns"],case .object(var first)=rows.first else {return XCTFail("Current preprocessing rules")}
        first["center"] = .bool(false);rows[0] = .object(first);config["columns"] = .array(rows)
        _ = try PreprocessConfiguration(values:config)
        document.nodes[index].payload = .dataOperator(operatorID:op,config:config)
        XCTAssertThrowsError(try validate(document,archive),"Archived fitted preprocessing is part of selected semantics")
    }
    func testChangedDependencyConnectionAndMissingArchiveSourceReject()throws {
        let (document,archive)=try fixture()
        var changed=archive
        let cast=try XCTUnwrap(changed.edges.firstIndex {$0.to.node=="cast"})
        changed.edges[cast].from.node="split"
        XCTAssertThrowsError(try validate(document,changed))
        changed=archive;changed.nodes.removeAll {$0.id=="observations"}
        XCTAssertThrowsError(try validate(document,changed))
    }
    func testCapturedPlanCannotBeReusedAfterDocumentRevisionOrEndpointChanges()throws {
        var (document,archive)=try fixture()
        let (model,data)=try registries()
        let plan=try WorkflowEvaluationCompiler.compile(document,evaluate:terminal,modelRegistry:model,dataRegistry:data)
        document.revision+=1
        XCTAssertThrowsError(try WorkflowEvaluationDeclaration.validate(document:document,plan:plan,archivedDataflow:archive))
        document.revision-=1
        let edge=try XCTUnwrap(document.edges.firstIndex {$0.to.address==terminal && $0.to.port=="test"})
        document.edges[edge].from.port="validation"
        XCTAssertThrowsError(try WorkflowEvaluationDeclaration.validate(document:document,plan:plan,archivedDataflow:archive))
    }
}
