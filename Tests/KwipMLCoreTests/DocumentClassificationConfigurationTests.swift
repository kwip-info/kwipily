import XCTest
@testable import KwipMLCore

final class DocumentClassificationConfigurationTests:XCTestCase {
    func testLegacyAndDocumentDefaultsRemainDistinct()throws {
        var numeric=WorkflowEstimatorConfiguration.sklearnDefaults
        numeric["featureColumns"] = .array([.string("x"),.string("y")]);numeric["targetColumn"] = .string("label")
        let legacy=try WorkflowEstimatorConfiguration.decode(numeric)
        XCTAssertFalse(legacy.isDocumentClassification)
        XCTAssertEqual(legacy.featureColumns,["x","y"])
        let document=try WorkflowEstimatorConfiguration.decode(DocumentClassificationConfiguration.defaults)
        XCTAssertTrue(document.isDocumentClassification);XCTAssertEqual(document.featureColumns,["text"])
        numeric["task"] = .string("unknown");XCTAssertThrowsError(try WorkflowEstimatorConfiguration.decode(numeric))
    }
    func testInvalidFeatureAndSolverChoicesRefuse()throws {
        try DocumentClassificationConfiguration.validateFeatures(DocumentClassificationConfiguration.featureDefaults)
        var c=DocumentClassificationConfiguration.featureDefaults;c["ngramMin"] = 4;c["ngramMax"] = 2
        XCTAssertThrowsError(try DocumentClassificationConfiguration.validateFeatures(c))
        c=DocumentClassificationConfiguration.featureDefaults;c["maximumDF"] = 0
        XCTAssertThrowsError(try DocumentClassificationConfiguration.validateFeatures(c))
        var settings=DocumentClassificationConfiguration.estimatorDefaults;settings["solver"] = .string("arbitrary")
        XCTAssertThrowsError(try DocumentClassificationConfiguration.validateEstimator(settings))
    }
    func testPracticeCompilesFitPredictSaveAndTrainingCodeWithoutChangingLegacy()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let doc=try DocumentClassificationPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
        let ports=try WorkflowPortContracts.resolve(doc.nodes.first {$0.address==DocumentClassificationPracticeExample.estimator}!,document:doc,modelRegistry:model,dataRegistry:data)
        XCTAssertEqual(ports.outputs.map(\.id),["model","predictions"])
        XCTAssertEqual(ports.outputs.last?.kind,.table)
        let fit=try WorkflowEstimatorCompiler.compile(doc,estimator:DocumentClassificationPracticeExample.estimator,modelRegistry:model,dataRegistry:data)
        XCTAssertTrue(fit.configuration.isDocumentClassification);XCTAssertFalse(fit.configuration.isPrediction)
        let predict=try WorkflowEstimatorCompiler.compile(doc,estimator:DocumentClassificationPracticeExample.predict,modelRegistry:model,dataRegistry:data)
        XCTAssertTrue(predict.configuration.isPrediction);XCTAssertEqual(predict.sourceAddresses,[DocumentClassificationPracticeExample.unseen])
        XCTAssertEqual(predict.documentModelSource,.init(domain:.control,localID:"load"))
        let saved=try WorkflowSaveModelPlan.compile(doc,save:.init(domain:.control,localID:"save"))
        XCTAssertTrue(saved.modelSource.isEstimator)
        let code=try WorkflowCodePlan.compile(doc,scope:.training,selectedTargets:[DocumentClassificationPracticeExample.estimator],modelRegistry:model,dataRegistry:data)
        XCTAssertEqual(code.trainings.first?.configuration?["task"],.string(DocumentClassificationConfiguration.task))
        XCTAssertThrowsError(try WorkflowCodePlan.compile(doc,scope:.training,selectedTargets:[DocumentClassificationPracticeExample.predict],modelRegistry:model,dataRegistry:data))
        let old=try WorkflowExamples.preparedSklearnClassifier()
        let legacy=try WorkflowEstimatorCompiler.compile(old,estimator:WorkflowExamples.sklearnEstimatorAddress,modelRegistry:model,dataRegistry:data)
        XCTAssertFalse(legacy.configuration.isDocumentClassification)
    }

}
