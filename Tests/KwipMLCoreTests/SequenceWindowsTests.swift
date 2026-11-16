import XCTest
@testable import KwipMLCore

final class SequenceWindowsTests:XCTestCase {
    private func registry()throws->DataFlowOperatorRegistry {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}
    func testConfigurationBoundsNamesAndConditionalOutputs()throws {
        let defaults=SequenceWindowsConfiguration.defaults
        try SequenceWindowsConfiguration.validate(defaults)
        XCTAssertEqual(SequenceWindowsConfiguration.columns(defaults).map(\.name),["window","document_id","start","parent_id","input_ids","target_ids","valid_length"])
        for (key,value):(String,JSONValue) in [("contextLength",0),("stride",true),("paddingID",65536),("inputColumn",.string("target_ids")),("maskColumn",.string("window")),("scope",.string("document"))] {
            var c=defaults;c[key]=value;XCTAssertThrowsError(try SequenceWindowsConfiguration.validate(c),key)
        }
        var padded=defaults;padded["incomplete"] = .string("pad");padded["maskColumn"] = .string("valid_targets")
        XCTAssertEqual(SequenceWindowsConfiguration.columns(padded).last?.name,"valid_targets")
        XCTAssertThrowsError(try SequenceWindowsConfiguration.validate(defaults,schema:[.init(name:"tokens",dtype:"float64")]))
        try SequenceWindowsConfiguration.validate(defaults,schema:[.init(name:"tokens",dtype:"int64Sequence")])
    }
    func testPracticePersistencePortsAndPreparationCode()throws {
        let document=try SequenceWindowsPracticeExample.document(),data=try registry(),models=try ComponentRegistry.builtIn()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        let node=try XCTUnwrap(document.nodes.first {$0.address==SequenceWindowsPracticeExample.address})
        let ports=try WorkflowPortContracts.resolve(node,document:document,modelRegistry:models,dataRegistry:data)
        XCTAssertEqual(ports.inputs.map(\.id),["table"]);XCTAssertEqual(ports.outputs.map(\.id),["table"])
        var partitioned=node
        var c=SequenceWindowsConfiguration.defaults;c["mode"] = .string("partitions")
        partitioned.payload = .dataOperator(operatorID:SequenceWindowsConfiguration.operatorID,config:c)
        let partitionPorts=try WorkflowPortContracts.resolve(partitioned,document:document,modelRegistry:models,dataRegistry:data)
        XCTAssertEqual(partitionPorts.inputs.map(\.id),["training","validation","test"])
        let plan=try WorkflowCodePlan.compile(document,scope:.preparation,modelRegistry:models,dataRegistry:data)
        XCTAssertTrue(plan.trainings.isEmpty)
    }
    func testTrainRefusesWindowsWithoutNextTokenMapping()throws {
        var doc=try TextEncodingPracticeExample.trainingDocument()
        var c=SequenceWindowsConfiguration.defaults;c["mode"] = .string("partitions");c["contextLength"]=5;c["stride"]=5
        doc.nodes.append(.init(address:SequenceWindowsPracticeExample.address,payload:.dataOperator(operatorID:SequenceWindowsConfiguration.operatorID,config:c)))
        for i in doc.edges.indices where doc.edges[i].from.address==TextEncodingPracticeExample.address && ["training","validation"].contains(doc.edges[i].from.port) {doc.edges[i].from.address=SequenceWindowsPracticeExample.address}
        for role in ["training","validation"] {doc.edges.append(.init(from:.init(address:TextEncodingPracticeExample.address,port:role),to:.init(address:SequenceWindowsPracticeExample.address,port:role),kind:.table))}
        XCTAssertThrowsError(try WorkflowCompiler.compile(doc,train:WorkflowExamples.trainAddress,modelRegistry:ComponentRegistry.builtIn(),dataRegistry:registry(),contractVersion:15)) {error in XCTAssertTrue(error.localizedDescription.contains("Next-token objective"),error.localizedDescription)}
    }
}
