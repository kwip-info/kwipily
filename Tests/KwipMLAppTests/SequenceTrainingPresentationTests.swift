import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SequenceTrainingPresentationTests:XCTestCase {
    func testNextTokenMappingAndObjectiveApplyAsOneUndoableTransaction()throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:directory)}
        var document=try SequenceTrainingPracticeExample.document()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        guard case .train(let target)=document.nodes[index].payload,case .object(let mapping)=target["dataMapping"] else {return XCTFail("Sequence mapping missing")}
        var start=target;start.removeValue(forKey:"objective");start["dataMapping"] = .object(["featureColumns":.array([.string("input_ids")]),"targetColumn":.string("target_ids")]);document.nodes[index].payload = .train(config:start)
        let library=ProjectLibrary(directory:directory);_=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let context=try WorkflowControlNodeContext(owner:owner,address:WorkflowExamples.trainAddress)
        let objective=try LossConfiguration(componentID:"loss_cross_entropy",values:[:])
        XCTAssertTrue(context.configureSequenceMapping(mapping,objective:objective,expected:start),context.statusMessage)
        XCTAssertEqual(context.configuration["dataMapping"],target["dataMapping"])
        XCTAssertEqual(context.configuration["objective"],target["objective"])
        XCTAssertFalse(context.configureSequenceMapping(mapping,objective:objective,expected:start))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes.first {$0.address==WorkflowExamples.trainAddress}?.payload,.train(config:start))
    }
    func testValidationPerplexityLabelsCountsAndOverflowRemainHonest()throws {
        let e=ObjectiveEvaluation(sequenceObjective:true,componentID:"loss_cross_entropy",observations:2,activeCount:3,ignoredCount:3,maskedCount:0,effectiveScalar:log(4),aggregationNumerator:3*log(4),aggregationDenominator:3,aggregationConvention:"nativeMean",rawShape:[],accuracy:0.5,rawPreview:nil)
        XCTAssertEqual(try XCTUnwrap(e.perplexity),4,accuracy:1e-12)
        let p=ObjectivePredictionPreview(runID:"test",source:WorkflowExamples.trainAddress,evaluationCount:2,rows:[],captureStatus:.omittedByPolicy)
        let summary=try XCTUnwrap(ObjectiveResultProjection.sections(preview:p,evaluation:e).first)
        XCTAssertTrue(summary.rows.contains {["Perplexity","4"]==$0})
        XCTAssertTrue(summary.rows.contains {["Valid tokens","3"]==$0})
        var legacy=e;legacy.sequenceObjective=false;XCTAssertNil(legacy.perplexity)
        let huge=ObjectiveEvaluation(sequenceObjective:true,componentID:"loss_cross_entropy",observations:1,activeCount:1,ignoredCount:0,maskedCount:0,effectiveScalar:1000,aggregationNumerator:1000,aggregationDenominator:1,aggregationConvention:"nativeMean",rawShape:[],accuracy:nil,rawPreview:nil)
        XCTAssertNil(huge.perplexity)
    }
}

extension SequenceTrainingPresentationTests {
    func testSequenceCanvasUsesItsConnectedObjectiveInsteadOfBinaryClassifierValidation()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:directory)}
        let d=try SequenceTrainingPracticeExample.document(),library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(d),id:d.projectID)
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:d.projectID,modelRegistry:models,dataRegistry:data)
        let adapter=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data);defer {adapter.invalidate()}
        for _ in 0..<200 {if !adapter.isInferringTensors {break};try await Task.sleep(for:.milliseconds(10))}
        XCTAssertFalse(adapter.isInferringTensors)
        XCTAssertNil(adapter.modelConfigurationIssue(for:.init(domain:.model,localID:"output")))
        let head=try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain:.model,localID:"head")])
        XCTAssertTrue(adapter.canvasPortDescriptor(nodeID:head,port:"out",direction:.output).declared.joined(separator:" ").contains("3, 4"))
    }
}
