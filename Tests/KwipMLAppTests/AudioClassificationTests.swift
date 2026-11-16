import XCTest
import KwipMLCore
@testable import KwipMLApp

final class AudioClassificationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testPracticeUsesRequiredModelEdgeAndStandardTableOutputs()throws {
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let doc=try AudioClassificationPracticeExample.document()
        let plan=try WorkflowDataPreparationPlan.compile(doc,registry:registry)
        XCTAssertEqual(plan.nodes.count,5);XCTAssertEqual(plan.edges.count,4)
        let node=try XCTUnwrap(doc.nodes.first {$0.address==AudioClassificationPracticeExample.classifier})
        let ports=try WorkflowPortContracts.resolve(node,document:doc,modelRegistry:try ComponentRegistry.builtIn(),dataRegistry:registry)
        XCTAssertEqual(ports.inputs.first {$0.id=="model"}?.kind,.frozenModel)
        XCTAssertEqual(ports.inputs.first {$0.id=="model"}?.required,true)
        XCTAssertEqual(Set(ports.outputs.map(\.id)),["predictions","scores"])
        if case .dataOperator(let id,let config)=node.payload {
            let definition=try XCTUnwrap(registry.operatorDefinition(id:id));XCTAssertEqual(config,definition.defaultConfig)
            try AudioClassificationConfiguration.validate(config,definition:definition)
            var broken=config;broken["hopSeconds"]=11
            XCTAssertThrowsError(try AudioClassificationConfiguration.validate(broken,definition:definition))
        }
        var disconnected=doc;disconnected.edges.removeAll {$0.to.port=="model"}
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(disconnected,registry:registry))
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
    }
    @MainActor func testLaunchHasNoImplicitSourceOrDownload()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(AudioClassificationPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertNil(host.bindings.path(for:AudioClassificationPracticeExample.source));XCTAssertNil(host.modelLoader.reference(AudioClassificationPracticeExample.model))
        XCTAssertEqual(ComponentLessonDestination.entryID(type:AudioClassificationConfiguration.operatorID),AudioClassificationPracticeExample.exampleID)
        let content=try XCTUnwrap(LessonContentCatalog.content(entryID:AudioClassificationPracticeExample.exampleID));XCTAssertGreaterThanOrEqual(content.steps.count,5)
    }
    func testHubASTInventoryNeedsProcessorWhileCausalStillNeedsTokenizer()throws {
        let files:[String]=["config.json","preprocessor_config.json","README.md","model.safetensors"]
        let raw:[String:Any]=["sha":"f826b80d28226b62986cc218e5cec390b1096902","config":["architectures":["ASTForAudioClassification"],"model_type":"audio-spectrogram-transformer"],"siblings":files.map{["rfilename":$0,"size":100]}]
        let config=LoadModelConfiguration(source:.huggingFace,repository:"MIT/ast-finetuned-audioset-10-10-0.4593")
        let data=try JSONSerialization.data(withJSONObject:raw)
        XCTAssertEqual(try ModelHubSnapshot.decode(data,configuration:config).files.count,4)
        var causal=raw;causal["config"]=["architectures":["GPT2LMHeadModel"],"model_type":"gpt2"]
        XCTAssertThrowsError(try ModelHubSnapshot.decode(JSONSerialization.data(withJSONObject:causal),configuration:config))
        var incomplete=raw;incomplete["siblings"]=files.filter{$0 != "preprocessor_config.json"}.map{["rfilename":$0,"size":100]}
        XCTAssertThrowsError(try ModelHubSnapshot.decode(JSONSerialization.data(withJSONObject:incomplete),configuration:config))
    }
    func testTaskSpecificSourceAdmissionPreservesTextTokenizer() {
        XCTAssertEqual(ModelAssetFiles.requiredSourceFiles(["model_type":.string("audio-spectrogram-transformer"),"architectures":.array([.string("ASTForAudioClassification")])]),["config.json","preprocessor_config.json"])
        XCTAssertEqual(ModelAssetFiles.requiredSourceFiles(["model_type":.string("gpt2")]),["config.json","tokenizer.json"])
    }
}
