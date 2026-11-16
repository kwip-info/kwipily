import XCTest
@testable import KwipMLCore

final class EmbeddingBagLearningTests:XCTestCase {
    func testIndependentRepeatedEmptyAndWeightedArithmetic()throws {
        let rows=LearningTensor(shape:[3,3],values:[1,2,1,0,0,0,3,2,0])
        let table=LearningTensor(shape:[4,2],values:[0,0,2,-1,1,3,-2,4])
        var values=EmbeddingBagConfiguration.defaults
        values["num_embeddings"]=4;values["embedding_dim"]=2;values["layout"] = .string("packed");values["mode"] = .string("sum")
        let reference=try EmbeddingBagLessonReference(sourceRows:rows,lengths:[3,0,1],table:table,itemWeights:nil,configuration:EmbeddingBagConfiguration(values:values))
        XCTAssertEqual(reference.output,LearningTensor(shape:[3,2],values:[5,1,0,0,-2,4]))
        let upstream=LearningTensor(shape:[3,2],values:[1,2,3,4,5,6])
        XCTAssertEqual(try reference.tableGradient(for:upstream),LearningTensor(shape:[4,2],values:[0,0,2,4,1,2,5,6]))
        values["per_sample_weights"]=true
        let weighted=try EmbeddingBagLessonReference(sourceRows:rows,lengths:[3,0,1],table:table,itemWeights:.init(shape:[3,3],values:[0.5,2,-1,9,8,7,2,3,4]),configuration:EmbeddingBagConfiguration(values:values))
        XCTAssertEqual(weighted.output,LearningTensor(shape:[3,2],values:[1,6.5,0,0,-4,8]))
        XCTAssertEqual(try weighted.itemWeightGradient(for:upstream),LearningTensor(shape:[3,3],values:[0,7,0,0,0,0,14,0,0]))
    }
    func testSixPlansArePrivateSeededAndComplete()throws {
        XCTAssertEqual(EmbeddingBagLearningTask.allCases.count,6)
        for task in EmbeddingBagLearningTask.allCases {
            let plan=try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"test",seed:17)
            try plan.validate()
            XCTAssertNotEqual(plan.fixtureDigest,try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"test",seed:18).fixtureDigest)
            XCTAssertFalse(plan.fixtures.contains {$0.expected["indicesGradient"] != nil || $0.expected["offsetsGradient"] != nil})
            let publicJSON=String(data:try JSONEncoder().encode(plan.publicTask(mode:.assessment)),encoding:.utf8)!
            XCTAssertFalse(publicJSON.contains("fixtureSeed"));XCTAssertFalse(publicJSON.contains("expected"))
            let graph=try EmbeddingBagExamples.lessonGraph(task:task)
            let submission=try EmbeddingBagLearningSubmission(graph:graph)
            XCTAssertEqual(try EmbeddingBagLearningSubmission.decode(submission.encoded()),submission)
            XCTAssertThrowsError(try TensorLearningSubmission(graph:graph))
            var invalid=graph;invalid.edges[0].sourcePort="out"
            XCTAssertThrowsError(try EmbeddingBagLearningSubmission(graph:invalid))
            var wrong=graph;wrong.nodes[0].type="input"
            XCTAssertThrowsError(try EmbeddingBagLearningSubmission(graph:wrong))
        }
    }
    func testSourceValidationRejectsUnusedInvalidCells()throws {
        let task=EmbeddingBagLearningTask.packedMean
        var rows=LearningTensor(shape:[2,3],values:[1,2,0,0,0,0]);rows.values[5]=4
        XCTAssertThrowsError(try EmbeddingBagLessonReference(sourceRows:rows,lengths:[2,0],table:.init(shape:[4,3],values:Array(repeating:1,count:12)),itemWeights:nil,configuration:task.configuration))
    }
    private func grade(_ plan:LearningAssessmentPlan,_ transform:(LearningFixture)->[String:LearningTensor])throws->LearningGradeStatus {
        try LearningAssessor.evaluate(plan,evidence:.init(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:String(repeating:"a",count:64),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:transform($0))},elapsedMilliseconds:1,maximumTensorElements:1000)).status
    }
    func testAssessorRequiresAllMeasurementsAndRejectsWrongInfluence()throws {
        for task in EmbeddingBagLearningTask.allCases {
            let plan=try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"test",seed:31)
            XCTAssertEqual(try grade(plan){$0.expected},.passed)
            XCTAssertEqual(try grade(plan){["output":$0.expected["output"]!]},.incomplete)
            XCTAssertEqual(try grade(plan){fixture in
                var measured=fixture.expected
                var influence=measured["tableGradient"]!
                influence.values=influence.values.map {$0+1}
                measured["tableGradient"]=influence
                return measured
            },.failed)
            if try task.configuration.usesPerSampleWeights {
                XCTAssertEqual(try grade(plan){fixture in
                    var measured=fixture.expected
                    var influence=measured["sourceWeightsGradient"]!
                    influence.values=influence.values.map {$0+1}
                    measured["sourceWeightsGradient"]=influence
                    return measured
                },.failed)
            }
        }
    }
    func testMeanAndCoordinateMaxIndependentExpectations()throws {
        let rows=LearningTensor(shape:[3,3],values:[1,2,1,0,0,0,3,2,0])
        let table=LearningTensor(shape:[4,2],values:[0,0,2,-1,1,3,-2,4])
        let upstream=LearningTensor(shape:[3,2],values:[1,2,3,4,5,6])
        for mode in ["mean","max"] {
            var config=EmbeddingBagConfiguration.defaults
            config["num_embeddings"]=4;config["embedding_dim"]=2
            config["layout"] = .string("packed");config["mode"] = .string(mode)
            let reference=try EmbeddingBagLessonReference(sourceRows:rows,lengths:[3,0,1],table:table,itemWeights:nil,configuration:EmbeddingBagConfiguration(values:config))
            let expected:[Double]=mode == "mean" ? [5.0/3,1.0/3,0,0,-2,4]:[2,3,0,0,-2,4]
            for (actual,wanted) in zip(reference.output.values,expected) {XCTAssertEqual(actual,wanted,accuracy:1e-12)}
            let gradient=try XCTUnwrap(reference.tableGradient(for:upstream))
            let wanted:[Double]=mode == "mean" ? [0,0,2.0/3,4.0/3,1.0/3,2.0/3,5,6]:[0,0,1,0,0,2,5,6]
            for (actual,expected) in zip(gradient.values,wanted) {XCTAssertEqual(actual,expected,accuracy:1e-12)}
        }
    }
    func testPaddingExcludedNormChangesSelectedRowsAndFreezeIsUnavailable()throws {
        var config=EmbeddingBagConfiguration.defaults
        config["num_embeddings"]=4;config["embedding_dim"]=2
        config["layout"] = .string("packed");config["mode"] = .string("sum")
        config["padding_idx"]=1;config["max_norm"]=1
        let rows=LearningTensor(shape:[2,3],values:[1,2,1,3,3,3])
        let table=LearningTensor(shape:[4,2],values:[9,12,3,4,0,2,-2,4])
        let upstream=LearningTensor(shape:[2,2],values:[2,3,4,5])
        let reference=try EmbeddingBagLessonReference(sourceRows:rows,lengths:[3,0],table:table,itemWeights:nil,configuration:EmbeddingBagConfiguration(values:config))
        XCTAssertEqual(reference.output.values[0],0)
        XCTAssertEqual(reference.output.values[1],2/(2+1e-7),accuracy:1e-12)
        XCTAssertEqual(Array(reference.tableAfterForward.values.prefix(2)),[9,12])
        XCTAssertEqual(reference.tableAfterForward.values[2],3/(5+1e-7),accuracy:1e-12)
        XCTAssertEqual(reference.tableAfterForward.values[3],4/(5+1e-7),accuracy:1e-12)
        XCTAssertEqual(Array(reference.tableAfterForward.values.suffix(2)),[-2,4])
        XCTAssertEqual(try reference.tableGradient(for:upstream),LearningTensor(shape:[4,2],values:[0,0,0,0,2,3,0,0]))
        config["freeze"]=true
        let frozen=try EmbeddingBagLessonReference(sourceRows:rows,lengths:[3,0],table:table,itemWeights:nil,configuration:EmbeddingBagConfiguration(values:config))
        XCTAssertEqual(frozen.tableAfterForward,reference.tableAfterForward)
        XCTAssertNil(try frozen.tableGradient(for:upstream))
    }

}
