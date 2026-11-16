import XCTest
@testable import KwipMLCore

final class TextGenerationTests:XCTestCase {
    func testSettingsRefuseMalformedBudgetsAndNames()throws {
        XCTAssertEqual(try TextGenerationConfiguration([:]).values["maxNewTokens"],.integer(64))
        for c:[String:JSONValue] in [["maxNewTokens":.integer(0)],["seed":.bool(true)],["topP":.number(0)],["textColumn":.string("generated_ids")],["device":.string("cuda")],["unknown":.integer(1)]] {XCTAssertThrowsError(try TextGenerationConfiguration(c))}
    }
    func testPracticeCodecAndArtifactSource()throws {
        let d=try TextGenerationPracticeExample.document()
        let copy=try JSONDecoder().decode(WorkflowDocument.self,from:JSONEncoder().encode(d))
        XCTAssertEqual(copy,d);try WorkflowCodec.validate(copy)
        XCTAssertEqual(try WorkflowTrainedModelSource.resolve(in:d,consumer:TextGenerationPracticeExample.address),.train(.init(address:WorkflowExamples.trainAddress,port:"model")))
        XCTAssertTrue(try TextGenerationConfiguration(["promptMode":.string("table")]).tableMode)
    }
}

extension TextGenerationTests {
    func testPromptConnectionCannotBeSilentlyRemovedAndTrainingStillCompiles()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),model=try ComponentRegistry.builtIn()
        var document=try TextGenerationPracticeExample.document()
        let i=try XCTUnwrap(document.nodes.firstIndex{$0.address==TextGenerationPracticeExample.address})
        document.nodes[i].payload = .generateText(config:["promptMode":.string("table"),"promptColumn":.string("text")])
        document.edges.append(.init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:TextGenerationPracticeExample.address,port:"prompts"),kind:.table))
        let ports=try WorkflowPortContracts.resolve(document.nodes[i],document:document,modelRegistry:model,dataRegistry:data)
        XCTAssertEqual(ports.inputs.map(\.id),["model","prompts"]);XCTAssertEqual(ports.outputs.map(\.id),["table"])
        XCTAssertThrowsError(try WorkflowEditing.apply(.configure(address:TextGenerationPracticeExample.address,config:["promptMode":.string("one")]),to:document,modelRegistry:model,dataRegistry:data))
        let training=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:15)
        XCTAssertEqual(training.namedInputPlan?.schemaVersion,2)
    }
}

extension TextGenerationTests {
    func testGeneratedTableBecomesIsolatedPreparationWithoutModelInputOwnership()throws {
        var document=try TextGenerationPracticeExample.document()
        let view=WorkflowNodeAddress(domain:.data,localID:"generated-view")
        document.nodes.append(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
        let endpoint=WorkflowEndpoint(address:TextGenerationPracticeExample.address,port:"table")
        document.edges.append(.init(from:endpoint,to:.init(address:view,port:"table"),kind:.table))
        let plan=try WorkflowPredictionPreparation(document:document,pins:[endpoint:String(repeating:"a",count:64)])
        XCTAssertNil(plan.document.namedModelInputs);XCTAssertNil(plan.document.modelInputs)
        XCTAssertEqual(Set(plan.sources.values),[endpoint]);try WorkflowCodec.validate(plan.document)
        XCTAssertTrue(plan.document.nodes.allSatisfy{$0.address.domain == .data})
    }
}

extension TextGenerationTests {
    func testWholeWorkflowCodeRefusesInsteadOfOmittingGeneration()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        XCTAssertThrowsError(try WorkflowCodePlan.compile(TextGenerationPracticeExample.document(),modelRegistry:.builtIn(),dataRegistry:data)) {error in
            XCTAssertTrue(error.localizedDescription.contains("Generate text"))
        }
    }
}
