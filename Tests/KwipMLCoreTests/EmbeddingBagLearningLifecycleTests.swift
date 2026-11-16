import XCTest
@testable import KwipMLCore

final class EmbeddingBagLearningLifecycleTests:XCTestCase {
    private func graph(_ task:EmbeddingBagLearningTask)throws->GraphDocument {
        try EmbeddingBagLearningSubmission(graph:EmbeddingBagExamples.lessonGraph(task:task)).graph
    }
    func testSixTaskRegistrationAndBlankDraftAdmission()throws {
        for task in EmbeddingBagLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue),task.rawValue)
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"test",seed:5),try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"test",seed:5))
            let blank=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided)
            XCTAssertNoThrow(try blank.validate())
            XCTAssertEqual(try TensorLearningDraft.decode(blank.encoded()),blank)
            XCTAssertThrowsError(try blank.submissionData())
        }
    }
    func testIncrementalConstructionPendingEditsAndReopen()throws {
        for task in EmbeddingBagLearningTask.allCases {
            let full=try graph(task)
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice)
            draft.graph.nodes=[full.nodes[0]]
            draft.notes="Each source row represents one observation group."
            draft.canvasView=CanvasViewState()
            draft.pendingFields=["source.maximum_items_per_bag":"unfinished"]
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submissionData())
            draft.graph.nodes.append(full.nodes[1]);draft.graph.edges=[full.edges[0]]
            XCTAssertNoThrow(try draft.validate())
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            draft.graph=full
            XCTAssertThrowsError(try draft.submissionData(),"Unapplied edits must block checking even after wiring is complete")
            draft.pendingFields=[:]
            let bytes=try draft.submissionData()
            XCTAssertEqual(try EmbeddingBagLearningSubmission.decode(bytes).graph,full)
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submission(),"Never reinterpret typed bag roles as an ordinary tensorGraph")
            draft.parameters=[:]
            XCTAssertThrowsError(try draft.validate(),"Private fixture-owned table values cannot be replaced by draft parameters")
        }
    }
    func testTemporaryConfigurationMismatchIsEditableButCannotSubmit()throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:EmbeddingBagLearningTask.fixedSum.rawValue,mode:.guided)
        draft.graph=try graph(.fixedSum)
        draft.graph.nodes[0].config["layout"] = .string("packed")
        XCTAssertNoThrow(try draft.validate(),"Allow changing source and operator controls one at a time")
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
        XCTAssertThrowsError(try draft.submissionData())
        draft.graph.nodes[1].config["layout"] = .string("packed")
        XCTAssertNoThrow(try draft.validate())
        XCTAssertThrowsError(try draft.submissionData(),"Packed construction still needs the offsets connection")
        draft.graph.edges.append(.init(id:"offsets",source:draft.graph.nodes[0].id,sourcePort:"offsets",target:draft.graph.nodes[1].id,targetPort:"offsets"))
        XCTAssertNoThrow(try draft.submissionData())
    }
    func testWrongRolesDuplicateSourcesAndOrdinaryInputsReject()throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:EmbeddingBagLearningTask.packedWeightedSum.rawValue,mode:.assessment)
        let full=try graph(.packedWeightedSum)
        draft.graph=full
        XCTAssertNoThrow(try draft.validate())
        draft.graph.edges[0].sourcePort="out"
        XCTAssertThrowsError(try draft.validate())
        draft.graph=full;draft.graph.nodes[0].type="input"
        XCTAssertThrowsError(try draft.validate())
        draft.graph=full;var duplicate=full.nodes[0];duplicate.id="other-source";draft.graph.nodes.append(duplicate)
        XCTAssertThrowsError(try draft.validate())
        draft.graph=full;draft.graph.nodes[0].config["per_sample_weights"]=false
        XCTAssertThrowsError(try draft.validate(),"A saved edge may not refer to a disabled source role")
        draft.graph=full;draft.graph.edges.append(full.edges[0])
        XCTAssertThrowsError(try draft.validate())
    }
    func testSixCanonicalReceiptsReplayAndRejectChangedEvidence()throws {
        for task in EmbeddingBagLearningTask.allCases {
            let plan=try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"test",seed:11)
            let bytes=try EmbeddingBagLearningSubmission(graph:graph(task)).encoded()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:1000)
            let receipt=try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice,submissionData:bytes,plan:plan,evidence:evidence)
            XCTAssertEqual(receipt.attempt.grade.status,.passed)
            XCTAssertEqual(try receipt.verifiedReplay(expectedProjectID:receipt.projectID),plan)
            let decoded=try TensorLearningReceipt.decode(receipt.encoded())
            XCTAssertEqual(decoded.submissionData,bytes)
            XCTAssertEqual(decoded.attempt,receipt.attempt)
            XCTAssertEqual(try decoded.verifiedReplay(expectedProjectID:receipt.projectID),plan)
            var changed=decoded;changed.fixtureSeed &+= 1
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
            changed=decoded;changed.evidence.fixtures[0].tensors["output"]!.values[0]+=1
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
            XCTAssertThrowsError(try decoded.verifiedReplay(expectedProjectID:UUID().uuidString))
        }
    }
}
