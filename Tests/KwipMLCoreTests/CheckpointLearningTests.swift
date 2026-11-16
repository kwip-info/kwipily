import XCTest
@testable import KwipMLCore
final class CheckpointLearningTests:XCTestCase {
    private func object(_ value:JSONValue)->[String:JSONValue] {guard case .object(let o)=value else {return [:]};return o}
    func testActualTaskRegistrationAndFourVariedBoundedPlans()throws {
        for task in CheckpointLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            let plan=try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:41)
            XCTAssertEqual(plan.fixtures.count,4);XCTAssertEqual(plan.budget.milliseconds,60_000)
            XCTAssertEqual(plan.runtime.capabilities,["checkpoint_lifecycle":ContractVersion()])
            XCTAssertEqual(Set(plan.fixtures.map {$0.inputs["trainingInputs"]!.values.map(String.init(describing:)).joined(separator:",")}).count,4)
        }
    }
    func testInitialAndIncompleteLifecycleDraftsAndOldDraftOmission()throws {
        let project=UUID().uuidString
        var draft=TensorLearningDraft(projectID:project,operation:CheckpointLearningTask.restoreWeights.rawValue,mode:.guided)
        XCTAssertEqual(draft.checkpointConfiguration?.trainingMode,.fresh)
        XCTAssertEqual(draft.checkpointConfiguration?.parent,CheckpointLearningParent.none)
        draft.checkpointConfiguration=try .init(trainingMode:.fineTune,parent:.none,epochs:1,batchSize:4)
        XCTAssertNoThrow(try draft.encoded())
        XCTAssertThrowsError(try draft.submissionData())
        var old=TensorLearningDraft(projectID:project,operation:"reshape",mode:.guided)
        XCTAssertNil(object(try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(old)))["checkpointConfiguration"])
        old.checkpointConfiguration=try .init()
        XCTAssertThrowsError(try old.validate())
    }
    func testStrictSubmissionAndIndependentScalarUpdate()throws {
        let task=CheckpointLearningTask.restoreWeights,raw=try CheckpointLearningSubmission(configuration:task.configuration,task:task).encoded()
        XCTAssertEqual(try CheckpointLearningSubmission.decode(raw,task:task).configuration,task.configuration)
        var object=self.object(try JSONDecoder().decode(JSONValue.self,from:raw))
        object["answers"] = .bool(true)
        XCTAssertThrowsError(try CheckpointLearningSubmission.decode(JSONEncoder().encode(JSONValue.object(object)),task:task))
        object.removeValue(forKey:"answers")
        var c=self.object(object["configuration"]!);c["epochs"] = .bool(true);object["configuration"] = .object(c)
        XCTAssertThrowsError(try CheckpointLearningSubmission.decode(JSONEncoder().encode(JSONValue.object(object)),task:task))
        let result=try CheckpointLessonReference.affineUpdate(parameters:[1,0,0,2,0,0],rows:Array(repeating:[1,0],count:8),targets:Array(repeating:[0,0],count:8))
        for (a,b) in zip(result.parameters,[0.77,0,-0.23,1.54,0,-0.46]) {XCTAssertEqual(a,b,accuracy:1e-12)}
        for (a,b) in zip(result.memory,[1.3,0,1.3,2.6,0,2.6]) {XCTAssertEqual(a,b,accuracy:1e-12)}
    }
    func testColdDraftAndCurrentReceiptRequireLifecycleEvidenceAndProtectPrivateValues()throws {
        for task in CheckpointLearningTask.allCases {
            let project=UUID().uuidString,plan=try CheckpointAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173)
            var draft=TensorLearningDraft(projectID:project,operation:task.rawValue,mode:.guided)
            var raw=try JSONEncoder().encode(draft),object=self.object(try JSONDecoder().decode(JSONValue.self,from:raw))
            object["checkpointConfiguration"] = try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(task.configuration))
            raw=try JSONEncoder().encode(JSONValue.object(object));draft=try TensorLearningDraft.decode(raw)
            let submission=try draft.submissionData()
            XCTAssertEqual(try CheckpointLearningSubmission.decode(submission,task:task).configuration,task.configuration)
            XCTAssertThrowsError(try draft.submission())
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map { f in
                var tensors=f.expected
                for (key,shape) in task.measuredShapes where tensors[key] == nil {tensors[key] = .init(shape:shape,values:Array(repeating:0,count:shape.reduce(1,*)))}
                return .init(fixtureID:f.id,tensors:tensors)
            },elapsedMilliseconds:1,maximumTensorElements:100)
            let receipt=try TensorLearningReceipt(projectID:project,operation:task.rawValue,mode:.guided,submissionData:submission,plan:plan,evidence:evidence)
            let cold=try TensorLearningReceipt.decode(receipt.encoded());XCTAssertNotNil(cold.checkpointFormativeProjection(expectedProjectID:project,currentSubmission:submission,currentMode:.guided))
            XCTAssertNil(cold.checkpointFormativeProjection(expectedProjectID:project,currentSubmission:submission,currentMode:.assessment))
            XCTAssertNil(cold.checkpointFormativeProjection(expectedProjectID:project,currentSubmission:Data(),currentMode:.guided))
            let privateReceipt=try TensorLearningReceipt(projectID:project,operation:task.rawValue,mode:.assessment,submissionData:submission,plan:plan,evidence:evidence)
            XCTAssertNil(privateReceipt.checkpointFormativeProjection(expectedProjectID:project,currentSubmission:submission,currentMode:.guided))
            var absentTable=receipt;absentTable.evidence.fixtures[0].tensors.removeValue(forKey:"savedParameters")
            XCTAssertThrowsError(try absentTable.verifiedReplay(expectedProjectID:project))
            XCTAssertNil(absentTable.checkpointFormativeProjection(expectedProjectID:project,currentSubmission:submission,currentMode:.guided))
            var missing=receipt;missing.evidence.fixtures[0].tensors.removeValue(forKey:plan.criteria[0].id)
            XCTAssertThrowsError(try missing.verifiedReplay(expectedProjectID:project));XCTAssertNil(missing.checkpointFormativeProjection(expectedProjectID:project,currentSubmission:submission,currentMode:.guided))
        }
    }
}
