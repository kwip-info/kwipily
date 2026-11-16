import XCTest
import KwipMLCore
@testable import KwipMLApp
final class LessonAudienceGuideTests: XCTestCase {
    func testEveryExistingTensorExerciseHasConcreteNoCodeGuidance() {
        for operation in TensorLearningTask.operations {
            let guide = LessonAudienceGuide.make(operation: operation)
            XCTAssertFalse(guide.goal.isEmpty, operation)
            if let task = CheckpointLearningTask(rawValue: operation) {
                XCTAssertTrue(guide.steps.contains { $0.contains(task.controlInstructions) }, operation)
                XCTAssertTrue(guide.steps.contains { $0.contains("Apply") }, operation)
                for name in ["Checkpoint", "Training memory", "Epoch", "Batch"] {
                    XCTAssertTrue(guide.terms.contains { $0.name == name }, operation+": "+name)
                }
                XCTAssertFalse(guide.terms.contains { $0.name == "Gradient" }, operation)
                continue
            }
            if let task = DataLoadingLearningTask(rawValue: operation) {
                XCTAssertTrue(guide.steps.contains { $0.contains(task.controlInstructions) }, operation)
                XCTAssertTrue(guide.steps.contains { $0.contains("Connect each out to the matching named Train input") }, operation)
                XCTAssertTrue(guide.terms.contains { $0.name == "Observation" }, operation)
                XCTAssertFalse(guide.terms.contains { $0.name == "Gradient" }, operation)
                continue
            }
            if let task = ProgramExportLearningTask(rawValue: operation) {
                XCTAssertTrue(guide.steps.contains { $0.contains(task.controlInstructions) }, operation)
                XCTAssertTrue(guide.steps.contains { $0.contains("Apply the artifact controls") }, operation)
                XCTAssertTrue(guide.steps.contains { $0.contains("reloaded results") }, operation)
                XCTAssertTrue(guide.terms.contains { $0.name == "Executable program" }, operation)
                XCTAssertTrue(guide.terms.contains { $0.name == "Batch" }, operation)
                XCTAssertFalse(guide.terms.contains { $0.name == "Gradient" }, operation)
                XCTAssertFalse(guide.steps.contains { $0.contains("Connect") }, operation)
                continue
            }
            XCTAssertTrue(guide.steps.contains { $0.contains("connect") || $0.contains("Connect") }, operation)
            if SchedulerLearningTask(rawValue: operation) != nil {
                for name in ["Learning rate", "Epoch", "Batch", "Scheduler"] {
                    XCTAssertTrue(guide.terms.contains { $0.name == name }, operation+": "+name)
                }
            } else if operation == LazyLinearAssessment.operation {
                XCTAssertTrue(guide.terms.contains { $0.name == "Initialization" }, operation)
            } else { XCTAssertTrue(guide.terms.contains { $0.name == "Gradient" }, operation) }
        }
    }
    func testTechnicalSpecificationsAndAssessmentAreNotRewritten() {
        let task = "Use [2,3] and 1.0. Preserve the batch axis."
        XCTAssertEqual(LessonAudienceGuide.requirements(task), task)
        XCTAssertTrue(LessonAudienceGuide.make(operation: "softmax-last-axis").goal.contains("add up to 1"))
    }

    func testEveryTensorExerciseHasApproachableRequirementsAndNoDerivativeAssignment() {
        for operation in TensorLearningTask.operations {
            let text = LessonAudienceGuide.requirements(operation: operation, fallback: "Technical fallback")
            XCTAssertNotEqual(text, "Technical fallback", operation)
            XCTAssertTrue(text.contains("**Configure**"), operation)
            if operation == LazyLinearAssessment.operation {
                XCTAssertTrue(text.contains("infers three input features"), operation)
            } else { XCTAssertTrue(text.contains("Check construction"), operation) }
            XCTAssertFalse(text.contains("derive"), operation)
            XCTAssertFalse(text.contains("L ="), operation)
            XCTAssertFalse(text.contains("Σ"), operation)
        }
    }

    func testLayoutRequirementsKeepPublicShapesAndDistinctValueOrder() {
        let reshape = requirements("reshape")
        XCTAssertTrue(reshape.contains("[2,3]"))
        XCTAssertTrue(reshape.contains("[3,2]"))
        XCTAssertTrue(reshape.contains("same reading order"))
        let transpose = requirements("transpose")
        XCTAssertTrue(transpose.contains("First axis to 1"))
        XCTAssertTrue(transpose.contains("Second axis to 2"))
        XCTAssertTrue(transpose.contains("rows become columns"))
        let flatten = requirements("flatten")
        XCTAssertTrue(flatten.contains("[6]"))
        XCTAssertTrue(flatten.contains("First axis to 1"))
        XCTAssertTrue(flatten.contains("Last axis to -1"))
    }

    func testSoftmaxDirectionsKeepObservationsSeparate() {
        XCTAssertTrue(requirements("softmax-last-axis").contains("Normalization axis to -1"))
        XCTAssertTrue(requirements("softmax-last-axis").contains("three values in each row"))
        XCTAssertTrue(requirements("softmax-first-sample-axis").contains("Normalization axis to 1"))
        XCTAssertTrue(requirements("softmax-first-sample-axis").contains("two values in each column"))
        for id in ["softmax-last-axis", "softmax-first-sample-axis"] {
            XCTAssertTrue(requirements(id).contains("Do not combine different observations"))
        }
    }

    func testLayerNormAffineExplainsRequiredParameterMeasurementCheckboxAndRepair() throws {
        let label = "Inspect this component’s parameter gradients"
        let instructions = LessonAudienceGuide.requirements(operation: "layer-norm-affine", fallback: "")
        XCTAssertTrue(instructions.contains(label))
        XCTAssertTrue(instructions.contains("Enable"))
        XCTAssertTrue(instructions.contains("[0.5,-1,2]"))
        XCTAssertTrue(instructions.contains("[1,-0.5,0.25]"))
        XCTAssertTrue(LessonAudienceGuide.make(operation: "layer-norm-affine").steps.joined(separator: " ").contains(label))
        for criterion in ["shared-scale-gradient", "shared-bias-gradient"] {
            let feedback = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: "layer-norm-affine", criterionID: criterion))
            XCTAssertTrue(feedback.nextStep.contains(label))
        }
        for operation in ["layer-norm-rows", "layer-norm-sample"] {
            XCTAssertFalse(LessonAudienceGuide.requirements(operation: operation, fallback: "").contains(label))
        }
    }
    func testLayerNormKeepsPublicConfigurationAndOptionalParameters() {
        for task in LayerNormLearningTask.allCases {
            let text = requirements(task.rawValue)
            XCTAssertTrue(text.contains(task.normalizedShape == [2,3] ? "[2,3]" : "[3]"))
            XCTAssertTrue(text.contains(String(task.epsilon)))
            if task == .affine {
                XCTAssertEqual(task.scale?.values, [0.5,-1,2])
                XCTAssertEqual(task.bias?.values, [1,-0.5,0.25])
                XCTAssertTrue(text.contains("[0.5,-1,2]"))
                XCTAssertTrue(text.contains("[1,-0.5,0.25]"))
            } else {
                XCTAssertTrue(text.contains("Disable learned scale and bias"))
            }
        }
    }

    func testEmbeddingKeepsPublicTableAndSpecialPolicies() throws {
        for task in EmbeddingLearningTask.allCases {
            let config = try task.configuration
            XCTAssertEqual(config.numberOfEmbeddings, 4)
            XCTAssertEqual(config.embeddingDimension, 3)
            XCTAssertTrue(requirements(task.rawValue).contains("4 rows"))
            XCTAssertTrue(requirements(task.rawValue).contains("3 values"))
        }
        XCTAssertTrue(requirements("embedding-padding-frequency").contains("Padding row to 0"))
        XCTAssertTrue(requirements("embedding-padding-frequency").contains("supplied values"))
        XCTAssertTrue(requirements("embedding-renormalize").contains("Maximum norm to 1"))
        XCTAssertTrue(requirements("embedding-renormalize").contains("Norm order to 2"))
    }

    func testMatrixAndBilinearDoNotRelaxPublicSharingRules() {
        XCTAssertTrue(requirements("matrix-contraction").contains("Do not transpose, scale or normalize"))
        XCTAssertTrue(requirements("matrix-broadcast-gradients").contains("equal sizes or size 1"))
        XCTAssertTrue(requirements("bilinear-with-bias").contains("Enable bias"))
        XCTAssertTrue(requirements("bilinear-without-bias").contains("Disable bias"))
        for id in ["bilinear-with-bias", "bilinear-without-bias"] {
            XCTAssertTrue(requirements(id).contains("match exactly"))
            XCTAssertTrue(requirements(id).contains("no size-1 expansion"))
        }
    }

    func testUnknownRequirementsRetainTheirSpecification() {
        XCTAssertEqual(LessonAudienceGuide.requirements(operation: "future-task", fallback: "Original public task"), "Original public task")
    }

    private func requirements(_ operation: String) -> String {
        LessonAudienceGuide.requirements(operation: operation, fallback: "Technical fallback")
    }
}

extension LessonAudienceGuideTests {
    func testEveryTensorCriterionHasReadableFeedbackWithoutDerivativeAssignments() throws {
        for operation in TensorLearningTask.operations {
            let runtimeVersion = RemainingActivationLearningTask(rawValue: operation)?.isDropout == true ? "2.14.1" : "presentation-test"
            let plan = try TensorLearningTask.plan(operation: operation, runtimeVersion: runtimeVersion, seed: 42)
            for criterion in plan.criteria {
                let feedback = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: operation, criterionID: criterion.id), "\(operation): \(criterion.id)")
                XCTAssertFalse(feedback.title.isEmpty)
                XCTAssertNotEqual(feedback.title, criterion.id)
                XCTAssertTrue(feedback.nextStep.contains("Check construction"), operation)
                for technical in ["Jacobian", "Σ", "∂", "dL/d", "diag(", "compute the derivative", "write code"] {
                    XCTAssertFalse(feedback.nextStep.contains(technical), "\(operation): \(criterion.id) includes \(technical)")
                }
            }
        }
    }
    func testSoftmaxFailedGradientRepairNamesTheTaskAxisInsteadOfGivingJacobian() throws {
        let rows = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: "softmax-last-axis", criterionID: "coupled-gradient"))
        let columns = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: "softmax-first-sample-axis", criterionID: "coupled-gradient"))
        XCTAssertEqual(rows.title, "Influence within each score group")
        XCTAssertTrue(rows.nextStep.contains("Normalization axis to -1"))
        XCTAssertTrue(columns.nextStep.contains("Normalization axis to 1"))
        XCTAssertTrue(rows.nextStep.contains("automatically"))
        XCTAssertNotEqual(rows.nextStep, columns.nextStep)
    }
    func testLookupAndNormalizationFeedbackKeepsExactPublicPolicyControls() throws {
        let lookup = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: "embedding-padding-frequency", criterionID: "tableGradient"))
        XCTAssertTrue(lookup.nextStep.contains("Padding row to 0"))
        XCTAssertTrue(lookup.nextStep.contains("Scale gradients by frequency"))
        let affine = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: "layer-norm-affine", criterionID: "shared-scale-gradient"))
        XCTAssertTrue(affine.nextStep.contains("learned scale and bias"))
        XCTAssertTrue(affine.nextStep.contains("parameter editor"))
    }
    func testUnknownFeedbackDoesNotMasqueradeAsReviewedCriterion() {
        XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: "future-task", criterionID: "output"))
        XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: "reshape", criterionID: "future-criterion"))
    }
}

extension LessonAudienceGuideTests {
    func testLossGuidanceRetainsRealControlsRolesAndVisiblePreparation() throws {
        for task in LossLearningTask.allCases {
            let text = requirements(task.rawValue)
            XCTAssertTrue(text.contains("Objective check"), task.rawValue)
            XCTAssertTrue(text.contains("**Configure**"), task.rawValue)
            XCTAssertTrue(text.contains("Input influence"), task.rawValue)
            XCTAssertTrue(text.contains("Combined score"), task.rawValue)
            XCTAssertEqual(text.contains("visible LogSoftmax"), task.usesLogSoftmax, task.rawValue)
            for mode in LossLearningAssessment.modes {
                let feedback = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: mode))
                XCTAssertTrue(feedback.nextStep.contains("Check construction"), task.rawValue)
            }
            XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: "invented-check"))
        }
        XCTAssertTrue(requirements(LossLearningTask.ctc.rawValue).contains("Blank class at 0"))
        XCTAssertTrue(requirements(LossLearningTask.triplet.rawValue).contains("0.8"))
        XCTAssertTrue(requirements(LossLearningTask.bce.rawValue).contains("Data connection"))
    }
    func testCheckpointGuidanceUsesActualSavedRunControlsWithoutInventedPorts() throws {
        for task in CheckpointLearningTask.allCases {
            let text = requirements(task.rawValue)
            XCTAssertTrue(text.contains(task.controlInstructions))
            XCTAssertTrue(text.contains("Apply the controls"))
            XCTAssertTrue(text.contains("Check construction"))
            XCTAssertTrue(text.contains("saved parent"))
            XCTAssertFalse(text.contains("Input →"))
            XCTAssertFalse(text.contains("gradient"))
            XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: "invented-check"))
        }
    }
}


extension LessonAudienceGuideTests {
    func testCompositionGuidanceUsesTransactionalCanvasActionsAndExplainsInfluence() {
        for task in ModuleCompositionLearningTask.allCases {
            let text=requirements(task.rawValue)
            XCTAssertTrue(text.contains("**Build**"),task.rawValue)
            XCTAssertTrue(text.contains("**Configure**"),task.rawValue)
            XCTAssertTrue(text.contains("Apply each Block draft"),task.rawValue)
            XCTAssertTrue(text.contains("Discard returns to the applied construction"),task.rawValue)
            XCTAssertTrue(text.contains("how a small change affects a result"),task.rawValue)
            XCTAssertTrue(text.contains("Private assessment keeps its measured tables hidden"),task.rawValue)
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertEqual(guide.goal,task.title)
            XCTAssertTrue(guide.steps.contains {$0.contains("Connect A to B")},task.rawValue)
            XCTAssertTrue(guide.terms.contains {$0.name == "Private copy"},task.rawValue)
            XCTAssertTrue(guide.terms.contains {$0.name == "Stored order"},task.rawValue)
            XCTAssertTrue(guide.terms.contains {$0.name == "Connected order"},task.rawValue)
        }
        XCTAssertTrue(requirements(ModuleCompositionLearningTask.buildReuse.rawValue).contains("fixed sign reversal and subtraction"))
        XCTAssertTrue(requirements(ModuleCompositionLearningTask.orderName.rawValue).contains("stored child order and names separately"))
    }
    func testCompositionFeedbackAdmitsOnlyItsPublicCriteriaWithoutAnswerValues() throws {
        for task in ModuleCompositionLearningTask.allCases {
            for criterionID in task.measuredShapes.keys {
                let feedback=try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:criterionID))
                XCTAssertEqual(feedback.title,ModuleCompositionLearningTask.label(criterionID))
                XCTAssertNotEqual(feedback.title,criterionID)
                XCTAssertTrue(feedback.nextStep.contains("Apply every pending Block draft"))
                XCTAssertTrue(feedback.nextStep.contains("Check construction"))
                XCTAssertTrue(feedback.nextStep.contains("do not enter calculated answers"))
                XCTAssertFalse(feedback.nextStep.contains("coefficients"))
                XCTAssertFalse(feedback.nextStep.contains("["))
            }
            XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"invented-check"))
        }
        XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation:ModuleCompositionLearningTask.buildReuse.rawValue,criterionID:"namedOutputs"))
        XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation:ModuleCompositionLearningTask.orderName.rawValue,criterionID:"privateAdjustment"))
    }
}
