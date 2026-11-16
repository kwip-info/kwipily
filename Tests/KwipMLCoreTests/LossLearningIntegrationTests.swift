import XCTest
@testable import KwipMLCore

final class LossLearningIntegrationTests:XCTestCase {
    private func evidence(_ plan:LearningAssessmentPlan,_ data:Data)->LearningEvidence {
        .init(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(data),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:128)
    }
    func testEightSharedDispatchesAndPublicPrivacyInAllThreeLearningModes()throws {
        for task in LossLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let expected=try LossLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:43)
            let actual=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:43)
            XCTAssertEqual(actual,expected)
            for mode in [LearningMode.guided,.practice,.assessment] {
                let publicData=try JSONEncoder().encode(actual.publicTask(mode:mode))
                let object=try XCTUnwrap(JSONSerialization.jsonObject(with:publicData) as? [String:Any])
                XCTAssertNil(object["fixtures"]);XCTAssertNil(object["fixtureSeed"]);XCTAssertNil(object["expected"])
                XCTAssertEqual(actual.publicTask(mode:mode).criteria.count,5)
            }
        }
    }
    func testStrictLossPlansRejectIdentityCriteriaRoleAndDynamicGeometryTampering()throws {
        for task in LossLearningTask.allCases {
            let valid=try LossLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:43)
            for mutation in 0..<13 {
                var p=valid
                switch mutation {
                case 0:p.objectiveID="unreviewed"
                case 1:p.contentVersion=ContractVersion(2)
                case 2:p.runtime.backend="unreviewed"
                case 3:p.runtime.capabilities=["another":ContractVersion()]
                case 4:p.criteria.removeLast()
                case 5:p.criteria[0].rule = .tensor(key:"evaluationObjective",absoluteTolerance:1e-6,relativeTolerance:1e-5)
                case 6:p.criteria[2].rule = .tensor(key:"targetContract",absoluteTolerance:1e-6,relativeTolerance:0)
                case 7:p.criteria[0].rule = .tensor(key:"trainingObjective",absoluteTolerance:0.5,relativeTolerance:0.5)
                case 8:p.fixtures[0].inputs["answer"] = .init(shape:[1],values:[1])
                case 9:p.fixtures[0].expected["extra"] = .init(shape:[1],values:[1])
                case 10:p.fixtures[0].expected["inputInfluence"] = .init(shape:[1],values:[1])
                case 11:p.fixtures[0].expected["reduction"] = .init(shape:[1],values:[1])
                default:p.fixtures[0].expected["targetContract"]!.values[0] += 1
                }
                XCTAssertThrowsError(try p.validate(),"\(task.rawValue) mutation \(mutation)")
            }
        }
        var old=try TensorLearningTask.plan(operation:"reshape",runtimeVersion:"2.14.1",seed:43)
        old.fixtures[0].inputs=Dictionary(uniqueKeysWithValues:(0..<17).map {("role\($0)",LearningTensor(shape:[1],values:[1]))})
        XCTAssertThrowsError(try old.validate())
    }
    func testDraftPartialSavePendingEditAndTypedSubmissionBoundaries()throws {
        for task in LossLearningTask.allCases {
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice)
            draft.graph=LossLearningCanvas.preparedGraph(task:task)
            XCTAssertNoThrow(try draft.validate())
            XCTAssertEqual(try draft.submissionData(),try LossLearningSubmission(graph:draft.graph,task:task).encoded())
            XCTAssertThrowsError(try draft.submission())
            draft.pendingFields["objective.reduction"]="sum"
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:];draft.graph.edges.removeLast()
            let cold=try TensorLearningDraft.decode(draft.encoded())
            XCTAssertEqual(cold.graph,try LossLearningSubmission.validatedGraph(draft.graph,task:task,complete:false));XCTAssertThrowsError(try cold.submissionData())
        }
    }
    func testLossReceiptColdReplayMeasuredFailuresAndQualifiedSourceStaleness()throws {
        for task in LossLearningTask.allCases {
            let plan=try LossLearningAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:43)
            let data=try LossLearningSubmission(graph:LossLearningCanvas.preparedGraph(task:task),task:task).encoded()
            for mode in [LearningMode.guided,.practice,.assessment] {
                let project=UUID().uuidString
                let receipt=try TensorLearningReceipt(projectID:project,operation:task.rawValue,mode:mode,submissionData:data,plan:plan,evidence:evidence(plan,data))
                let cold=try TensorLearningReceipt.decode(receipt.encoded())
                XCTAssertEqual(cold.attempt.grade.status,.passed);XCTAssertEqual(cold.attempt.grade.outcomes.count,5)
                XCTAssertThrowsError(try cold.verifiedReplay(expectedProjectID:UUID().uuidString))
                var changed=cold;changed.fixtureSeed += 1;XCTAssertThrowsError(try changed.encoded())
                changed=cold;changed.evidence.fixtures[0].tensors.removeValue(forKey:"inputInfluence");XCTAssertThrowsError(try changed.encoded())
                changed=cold;changed.evidence.fixtures[0].partitions["privateAnswers"]=["row"];XCTAssertThrowsError(try changed.encoded())
            }
            var wrong=evidence(plan,data);wrong.fixtures[0].tensors["trainingObjective"]!.values[0] += 10
            let failed=try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice,submissionData:data,plan:plan,evidence:wrong)
            XCTAssertEqual(failed.attempt.grade.status,.failed)
            XCTAssertEqual(failed.attempt.grade.outcomes.filter {!$0.passed}.map(\.criterionID),["training-objective"])
            var object=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
            object["objectiveSource"]=["domain":"model","localID":"objective"]
            let forged=try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys,.withoutEscapingSlashes])
            XCTAssertThrowsError(try LossLearningSubmission.decode(forged,task:task))
            XCTAssertThrowsError(try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided,submissionData:forged,plan:plan,evidence:evidence(plan,forged)))
        }
    }
}
