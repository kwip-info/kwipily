import XCTest
@testable import KwipMLCore

final class UnionContractTests:XCTestCase {
    func testTrainingCompilerCarriesUnionSourcesAndRefusesPartitionMixing()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        var document=try WorkflowExamples.preparedJoinedClassifier(device:"cpu")
        let address=WorkflowNodeAddress(domain:.data,localID:"union")
        let index=try XCTUnwrap(document.edges.firstIndex {$0.from.address.localID == "join" && $0.to.address.localID == "cast"})
        let previous=document.edges.remove(at:index)
        document.nodes.append(.init(address:address,payload:.dataOperator(operatorID:"kwip.data.union",config:UnionConfiguration.defaults)))
        for port in ["table1","table2"] {document.edges.append(.init(from:previous.from,to:.init(address:address,port:port),kind:.table))}
        document.edges.append(.init(from:.init(address:address,port:"table"),to:previous.to,kind:.table))
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(Set(plan.sourceAddresses),[WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress])
        let i=try XCTUnwrap(document.edges.firstIndex {$0.to.address == address})
        let incoming=document.edges.remove(at:i),split=WorkflowNodeAddress(domain:.data,localID:"early-split")
        document.nodes.append(.init(address:split,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25])))
        document.edges.append(.init(from:incoming.from,to:.init(address:split,port:"table"),kind:.table))
        document.edges.append(.init(from:.init(address:split,port:"training"),to:incoming.to,kind:.table))
        XCTAssertThrowsError(try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry)) {error in XCTAssertTrue(error.localizedDescription.contains("before partitioning"),error.localizedDescription)}
    }
    func testPortIdentityOrderAndCodec()throws {
        var c=try UnionConfiguration(UnionConfiguration.defaults)
        c.inputs=["table9","table2","table1"]
        XCTAssertEqual(c.nextPort(),"table10")
        XCTAssertEqual(try UnionConfiguration(c.values),c)
        for bad in [["table1"],["table1","table1"],["table0","table2"],["bad","table2"],["table1\n","table2"]] {var values=c.values;values["inputs"] = .array(bad.map(JSONValue.string));XCTAssertThrowsError(try UnionConfiguration(values))}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let definition=try XCTUnwrap(registry.operatorDefinition(id:"kwip.data.union"))
        XCTAssertEqual(definition.configured(c.values).inputs.map(\.id),c.inputs)
        XCTAssertEqual(definition.defaultConfig,UnionConfiguration.defaults)
        let document=try UnionPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        XCTAssertEqual(document.edges.count,4)
        XCTAssertEqual(UnionPracticeExample.sourceResources.count,3)
        for resource in UnionPracticeExample.sources {XCTAssertTrue(FileManager.default.fileExists(atPath:try UnionPracticeExample.resourceURL(resource).path))}
    }
}
