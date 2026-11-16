import XCTest
@testable import KwipMLCore

/// Current teaching seeds are new authoring, distinct from archived historical projects.
final class CurrentPracticeSeedTests: XCTestCase {
    private let retired:Set<String>=["kwip.data.cast","kwip.data.selectRename","kwip.data.filter","kwip.data.filterExpression","kwip.data.regexMatch","kwip.data.regexExtract","kwip.data.regexReplace","kwip.data.calculate","kwip.data.missing","kwip.data.split","kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.standardScalerFit","kwip.data.imputerFit","kwip.data.oneHotEncoderFit","kwip.data.transform"]
    func testCurrentPreparedPracticeGraphsUseOnlyCurrentToolsAndOwnedTrainingSettings()throws {
        var documents:[WorkflowDocument]=[
            try WorkflowExamples.preparedParallelClassifier(device:"cpu"),
            try WorkflowExamples.preparedEmbeddingClassifier(device:"cpu"),
            try PreparationPracticeExample.document(device:"cpu"),
            try WorkflowExamples.preparedGroupSplitClassifier(device:"cpu"),
            try WorkflowExamples.preparedTemporalSplitClassifier(device:"cpu"),
            try WorkflowExamples.preparedSklearnClassifier(),
            try WorkflowExamples.preparedTensorFlowClassifier()]
        documents += try OptimizerConfiguration.componentIDs.sorted().map {try WorkflowExamples.preparedOptimizerClassifier(componentID:$0,device:"cpu")}
        documents += try SchedulerConfiguration.componentIDs.sorted().map {try WorkflowExamples.preparedSchedulerClassifier(componentID:$0,device:"cpu")}
        documents += try LossLearningTask.allCases.map {try WorkflowExamples.preparedLossObjective(task:$0,device:"cpu")}
        documents += try GradientLearningTask.allCases.map {try GradientWorkflowExamples.document(task:$0,device:"cpu")}
        documents += [try WorkflowExamples.preparedInitializationStartingValues(device:"cpu"),try WorkflowExamples.preparedInitializationStartingSpread(device:"cpu")]
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        for document in documents {
            try WorkflowCodec.validate(document)
            if let train=document.nodes.first(where: {if case .train(let config)=$0.payload {return config["trainingMode"] != .string("fineTune")};return false}) {
                let plan=try WorkflowCompiler.compile(document,train:train.address,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13)
                XCTAssertTrue(Set(plan.dataflow.nodes.map(\.operatorID)).isDisjoint(with:retired),document.name)
            }
            for node in document.nodes {
                if case .dataOperator(let id,_) = node.payload {
                    XCTAssertFalse(retired.contains(id),document.name+": "+id)
                    if id == "kwip.data.view" {XCTAssertFalse(document.edges.contains {$0.from.address == node.address},"View is an inspection sink: "+document.name)}
                }
                switch node.payload {
                case .optimizerRecipe,.schedulerRecipe,.lossRecipe,.initializationRecipe,.gradientPolicyRecipe,.trainabilityRecipe,.datasetRecipe,.batchRecipe:
                    XCTFail("Current practice contains a retired configuration block: "+document.name)
                default:break
                }
            }
            XCTAssertFalse(document.edges.contains {["optimizer","scheduler","objective","initialization","gradientPolicy","trainability","dataset","loader"].contains($0.to.port) && $0.to.address == WorkflowExamples.trainAddress},document.name)
        }
    }
    func testBasePracticeDeliberatelyAuthorsThreePopulationsAndTrainingOnlyPreparation()throws {
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let partition=try XCTUnwrap(document.nodes.first {$0.address.localID=="split" && $0.address.domain == .data})
        guard case .dataOperator(let operatorID,let config)=partition.payload else {return XCTFail("Missing Partition")}
        XCTAssertEqual(operatorID,"kwip.data.partition")
        XCTAssertEqual(config["validationPercent"],25);XCTAssertEqual(config["testPercent"],25)
        let preparation=try XCTUnwrap(document.nodes.first {$0.address.localID=="fit" && $0.address.domain == .data})
        guard case .dataOperator(let preparationID,let settings)=preparation.payload else {return XCTFail("Missing Preprocess")}
        XCTAssertEqual(preparationID,"kwip.data.preprocess")
        let rules=try PreprocessConfiguration(values:settings)
        XCTAssertEqual(rules.mode,"partitions")
        XCTAssertEqual(rules.columns.map(\.column),["signal","context"])
        XCTAssertTrue(rules.columns.allSatisfy {$0.scaling=="standard"})
        XCTAssertEqual(Set(document.edges.filter {$0.to.address==preparation.address}.map {$0.to.port}),["training","validation","test"])
        let mapping=WorkflowNodeAddress(domain:.control,localID:"mapping")
        XCTAssertEqual(Set(document.edges.filter {$0.to.address == mapping}.map {$0.from.address}),[preparation.address])
        XCTAssertEqual(Set(document.edges.filter {$0.to.address == mapping}.map {$0.from.port}),["training","validation"])

    }
    func testGuidedCourseKeepsIdentityWithCurrentVersionAndCurrentControls()throws {
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        XCTAssertEqual(plan.id,"kwip.course.universal-classifier")
        XCTAssertEqual(plan.version,5)
        XCTAssertEqual(Set(plan.steps.map(\.address)),Set(plan.completeDocument.nodes.map(\.address)))
        XCTAssertTrue(plan.steps.first {$0.address.localID=="cast"}?.title.contains("Columns") == true)
        XCTAssertTrue(plan.steps.first {$0.address.localID=="fit"}?.title.contains("Preprocess") == true)
    }
    func testObjectiveSeedDeclaresMappingSchemaBeforeTrainCanonicalizationAndRetainsModelVersion()throws {
        for task in LossLearningTask.allCases {
            let document=try WorkflowExamples.preparedLossObjective(task:task,device:"cpu")
            XCTAssertGreaterThanOrEqual(document.schemaVersion,13)
            XCTAssertEqual(document.modelSourceVersions?["model"],2,task.rawValue)
            let train=try XCTUnwrap(document.nodes.first {$0.address==WorkflowExamples.trainAddress})
            guard case .train(let config)=train.payload else {return XCTFail("Missing Train")}
            XCTAssertEqual(try LossConfiguration.decodeRecipe(try XCTUnwrap(config["objective"])).componentID,task.componentID)
            XCTAssertTrue(document.nodes.contains {node in
                guard case .trainingData(let fields)=node.payload else {return false}
                return fields["lossTargetMapping"] != nil
            })
        }
    }

}
