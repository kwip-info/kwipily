import XCTest
@testable import KwipMLCore

final class DataLoadingLearningTests:XCTestCase {
    private var rows:[String:LearningTensor] {
        var source:[Double]=[],targets:[Double]=[],ids:[Double]=[]
        for row in 0..<9 {
            source.append(Double(row*2+1));source.append(Double(row*2+2))
            targets.append(Double(100+row));ids.append(Double(row))
        }
        return ["sourceRows":LearningTensor(shape:[9,2],values:source),
                "desiredRows":LearningTensor(shape:[9,1],values:targets),
                "rowIDs":LearningTensor(shape:[9],values:ids)]
    }
    func testFourSourceReferencesHandIndexOriginalRowsAndKeepDesiredOutcomesTogether()throws {
        for task in [DataLoadingLearningTask.indexed,.tensor,.subset,.iterable] {
            let reference=try DataLoadingLessonReference(task:task,inputs:rows)
            let selected=task == .subset ? [4,1,0]:Array(0..<(task == .iterable ? 5:9))
            XCTAssertEqual(reference.expected["selectedRowIDs"],.init(shape:[selected.count],values:selected.map(Double.init)))
            XCTAssertEqual(reference.expected["selectedInputs"]?.values,selected.flatMap {[Double($0*2+1),Double($0*2+2)]})
            XCTAssertEqual(reference.expected["selectedTargets"]?.values,selected.map {Double(100+$0)})
            XCTAssertEqual(reference.expected["sourceKind"]?.values,[Double([DataLoadingLearningTask.indexed,.tensor,.subset,.iterable].firstIndex(of:task)!)])
        }
    }
    func testHandGroupedRowsKeepShortFinalGroupAndExplicitPaddingMask()throws {
        let reference=try DataLoadingLessonReference(task:.batches,inputs:rows)
        XCTAssertEqual(reference.expected["batchRowIDs"],.init(shape:[3,4],values:[0,1,2,3,4,5,6,7,8,-1,-1,-1]))
        XCTAssertEqual(reference.expected["batchLengths"],.init(shape:[3],values:[4,4,1]))
        XCTAssertEqual(reference.expected["batchTargets"],.init(shape:[3,4],values:[100,101,102,103,104,105,106,107,108,0,0,0]))
        XCTAssertEqual(reference.expected["dropLastRowIDs"],.init(shape:[2,4],values:[0,1,2,3,4,5,6,7]))
    }
    func testEightNoCodeExamplesRoundTripAndRejectIncorrectOrIncompleteWires()throws {
        for task in DataLoadingLearningTask.allCases {
            let graph=try DataLoadingExamples.lessonGraph(task:task)
            let submission=try DataLoadingLearningSubmission(task:task,graph:graph)
            XCTAssertEqual(try DataLoadingLearningSubmission.decode(submission.encoded(),task:task),submission)
            XCTAssertEqual(Set(graph.nodes.map(\.type)),["dataset_source","data_loader","train"])
            XCTAssertEqual(Set(graph.edges.map(\.targetPort)),["dataset","loader"])
            var broken=graph;broken.edges.removeLast()
            XCTAssertThrowsError(try DataLoadingLearningSubmission(task:task,graph:broken))
            XCTAssertNoThrow(try DataLoadingLearningSubmission(task:task,graph:broken,complete:false))
            broken=graph;broken.edges[0].targetPort="model"
            XCTAssertThrowsError(try DataLoadingLearningSubmission(task:task,graph:broken,complete:false))
            XCTAssertNoThrow(try DataLoadingLearningSubmission(task:task,graph:.init(name:"Blank",nodes:[],edges:[]),complete:false))
        }
    }
    func testEightPrivatePlansBindIndependentReferencesAndRejectAlteredExpectedAnswers()throws {
        for task in DataLoadingLearningTask.allCases {
            let plan=try DataLoadingAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:19)
            try DataLoadingAssessment.validate(plan)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            XCTAssertEqual(plan.fixtures.count,4)
            for fixture in plan.fixtures {XCTAssertEqual(fixture.expected,try DataLoadingLessonReference(task:task,inputs:fixture.inputs).expected)}
            var altered=plan
            let key=try XCTUnwrap(altered.fixtures[0].expected.keys.sorted().first)
            altered.fixtures[0].expected[key]?.values[0] += 1
            XCTAssertThrowsError(try DataLoadingAssessment.validate(altered))
        }
    }
    func testSourceAlignmentAndPrivateFixtureShapesRefuseMissingOrMalformedValues()throws {
        var invalid=rows;invalid["desiredRows"] = .init(shape:[8,1],values:Array(repeating:0,count:8))
        XCTAssertThrowsError(try DataLoadingLessonReference(task:.indexed,inputs:invalid))
        invalid=rows;invalid["rowIDs"] = .init(shape:[9],values:Array(repeating:0,count:9))
        XCTAssertThrowsError(try DataLoadingLessonReference(task:.subset,inputs:invalid))
        invalid=rows;invalid["rowIDs"] = nil
        XCTAssertThrowsError(try DataLoadingLessonReference(task:.indexed,inputs:invalid))
    }
}
