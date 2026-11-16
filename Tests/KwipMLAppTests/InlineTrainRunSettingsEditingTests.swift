import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InlineTrainRunSettingsEditingTests:XCTestCase {
    private let train=WorkflowNodeAddress(domain:.control,localID:"train")
    private let other=WorkflowNodeAddress(domain:.control,localID:"other")
    private func fixture()throws->(ProjectLibrary,WorkflowAuthoringStore,WorkflowControlNodeContext) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let config:[String:JSONValue]=["runSetup":.object([:]),"trainingMode":.string("fresh"),"batching":.object(DataLoaderConfiguration.defaults)]
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:train,payload:.train(config:config)),
            .init(address:other,payload:.train(config:["runSetup":.object(["epochs":9])]))])
        document.schemaVersion=16
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[train]),lease:owner.lease()))
        return(library,owner,try WorkflowControlNodeContext(owner:owner,address:train))
    }
    func testDefaultsStayRawAndRunFieldsPreserveOtherTrainControlsAndUndo()throws {
        let(library,owner,context)=try fixture(),before=owner.document
        XCTAssertEqual(context.runConfiguration,[:]);XCTAssertEqual(context.preparationCachePolicy,.off)
        XCTAssertFalse(owner.canUndo)
        let expected=context.configuration
        XCTAssertFalse(context.setRunValue("epochs",value:.number(1.5),expected:expected))
        XCTAssertEqual(owner.document,before)
        XCTAssertTrue(context.setRunValue("epochs",value:7,expected:expected))
        XCTAssertEqual(context.runConfiguration,["epochs":7])
        XCTAssertEqual(context.configuration["batching"],expected["batching"])
        XCTAssertEqual(context.configuration["trainingMode"],expected["trainingMode"])
        XCTAssertEqual(owner.document.nodes[1],before.nodes[1])
        XCTAssertEqual(try library.loadSource(id:owner.projectID),.workflow(owner.document))
        XCTAssertFalse(context.setRunValue("seed",value:8,expected:expected),"A stale full Train snapshot cannot overwrite another edit")
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,before.nodes)
        XCTAssertFalse(context.isCurrent)
    }
    func testCacheAndExperimentsAreTrainOwnedAndRespectPendingBusyAndStaleAuthority()throws {
        let(_,owner,context)=try fixture(),before=owner.document
        let original=context.configuration,origin=context.numericOrigin(key:"epochs")
        context.setNumericPending(origin:origin,key:"epochs",pending:true)
        XCTAssertFalse(context.setPreparationCachePolicy(.reuse,expected:original))
        XCTAssertFalse(owner.setPreparationCachePolicy(.reuse,for:train,lease:owner.lease()))
        XCTAssertEqual(owner.document,before)
        context.setNumericPending(origin:origin,key:"epochs",pending:false)
        XCTAssertTrue(context.setPreparationCachePolicy(.materialize,expected:original))
        XCTAssertEqual(context.runConfiguration?["preparationCachePolicy"],.string("materialize"))
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[3,4])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        let expected=try XCTUnwrap(context.experimentConfiguration)
        XCTAssertEqual(context.experimentTrainNodes.map(\.address),[train])
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.setRunValue("epochs",value:9,expected:context.configuration))
        XCTAssertFalse(context.setExperimentSpecification(spec,expected:expected))
        owner.setExecutionBusy(false)
        XCTAssertTrue(context.setExperimentSpecification(spec,expected:expected))
        XCTAssertEqual(context.experimentSpecification,spec)
        XCTAssertEqual(context.configuration["batching"],original["batching"])
        XCTAssertEqual(owner.document.nodes[1],before.nodes[1])
        XCTAssertFalse(context.setExperimentSpecification(nil,expected:expected))
        XCTAssertTrue(context.setExperimentSpecification(nil,expected:try XCTUnwrap(context.experimentConfiguration)))
        XCTAssertNil(context.runConfiguration?["experiment"])
    }
    func testGradientCaptureUsesNestedSettingsWithoutChangingRunValuesOrSiblingTrain()throws {
        let(_,owner,context)=try fixture(),sibling=owner.document.nodes[1]
        XCTAssertTrue(context.setRunValue("seed",value:17,expected:context.configuration))
        let capture=GradientCaptureConfiguration(everyEpochs:2,maxBatchesPerEpoch:3,maxCaptures:12)
        XCTAssertTrue(context.setGradientCapture(capture,expected:context.configuration))
        XCTAssertEqual(context.gradientConfiguration,capture)
        XCTAssertEqual(context.runConfiguration?["seed"],17)
        XCTAssertNil(context.configuration["gradientCapture"])
        XCTAssertTrue(context.setGradientCapture(nil,expected:context.configuration))
        XCTAssertNil(context.gradientConfiguration)
        XCTAssertEqual(owner.document.nodes[1],sibling)
    }
    func testEvaluationRoleSelectorKeepsTestDefaultEncodingAndPreservesSettings()throws {
        let(library,original,_)=try fixture()
        let evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        var document=original.document
        document.nodes.append(.init(address:evaluate,payload:.evaluateTest(config:[:])))
        _=try library.saveSource(.workflow(document),id:original.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:original.modelRegistry,dataRegistry:original.dataRegistry)
        XCTAssertTrue(owner.select(.init(nodeAddresses:[evaluate]),lease:owner.lease()))
        let context=try WorkflowControlNodeContext(owner:owner,address:evaluate)
        XCTAssertTrue(context.setEvaluationValue("batchSize",value:17,expected:context.configuration))
        let testConfig=context.configuration
        XCTAssertTrue(context.setEvaluationPartition("validation",expected:testConfig))
        XCTAssertEqual(context.evaluationSettings?.partitionRole,"validation")
        XCTAssertEqual(context.configuration["settingsVersion"],2)
        XCTAssertEqual(context.configuration["batchSize"],17)
        XCTAssertFalse(context.setEvaluationPartition("test",expected:testConfig))
        XCTAssertFalse(context.setEvaluationPartition("training",expected:context.configuration))
        XCTAssertTrue(context.setEvaluationPartition("test",expected:context.configuration))
        XCTAssertEqual(context.configuration,testConfig)
    }
}
