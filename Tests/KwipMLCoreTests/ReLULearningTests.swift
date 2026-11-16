import XCTest
@testable import KwipMLCore

final class ReLULearningTests: XCTestCase {
    func testIndependentReferenceClipsNegativesPreservesPositivesAndRefusesNonfinite() throws {
        let reference = LearningExamples.reluReference
        for (x, expected) in [(-2.0, 0.0), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)] {
            XCTAssertEqual(try reference.apply(x), expected, accuracy: 1e-14)
            XCTAssertEqual(try reference.apply(reference.apply(x)), expected)
        }
        XCTAssertEqual(try reference.apply(-1000), 0)
        XCTAssertEqual(try reference.apply(1000), 1000)
        XCTAssertThrowsError(try reference.apply(.nan)); XCTAssertThrowsError(try reference.apply(.infinity))
        XCTAssertTrue(reference.instructions.contains("negative"))
    }
    func testVersionedTaskSeedReplayAssessmentSecrecyAndMisconceptions() throws {
        let identity = ModelConstructionLessonIdentity.relu
        try identity.validate()
        XCTAssertEqual(try JSONDecoder().decode(ModelConstructionLessonIdentity.self, from: JSONEncoder().encode(identity)), identity)
        let task = try ModelConstructionLessonTasks.publicTask(identity: identity)
        XCTAssertEqual(task.allowedNodeTypes, ["input", "linear", "relu", "add", "concat", "output"])
        let plan = try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 42)
        XCTAssertEqual(plan.fixtureDigest, try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 42).fixtureDigest)
        XCTAssertNotEqual(plan.fixtureDigest, try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 43).fixtureDigest)
        XCTAssertEqual(plan.fixtures.count, 4)
        XCTAssertTrue(plan.fixtures.allSatisfy { $0.inputs["x"]?.shape == [2,2] && $0.expected["output"]?.shape == [2,2] })
        XCTAssertEqual(plan.fixtures[0].inputs["x"]?.values.first, 0)
        XCTAssertEqual(plan.fixtures[1].inputs["x"]?.values.first, 0)
        XCTAssertNotEqual(plan.fixtures[0].inputs["x"]?.values[1], plan.fixtures[1].inputs["x"]?.values[1])
        XCTAssertEqual(plan.runtime.capabilities, ["linear": ContractVersion(1,1,0), "relu": ContractVersion(1,1,0)])
        let publicAssessment = String(decoding: try JSONEncoder().encode(plan.publicTask(mode: .assessment)), as: UTF8.self)
        for fixture in plan.fixtures { XCTAssertFalse(publicAssessment.contains(fixture.id)) }
        XCTAssertFalse(publicAssessment.contains("expected"))
        XCTAssertThrowsError(try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "test", fixtureSeed: 42, assessmentVersion: ContractVersion(9,0,0)))
        func grade(_ transform: (Double) throws -> Double) throws -> LearningGrade {
            let evidence = try plan.fixtures.map { fixture in
                LearningFixtureEvidence(fixtureID: fixture.id, tensors: ["output": .init(shape: [2,2], values: try fixture.inputs["x"]!.values.map(transform))])
            }
            return try LearningAssessor.evaluate(plan, evidence: .init(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion, fixtureDigest: plan.fixtureDigest, runtime: plan.runtime, submissionDigest: String(repeating: "a", count: 64), fixtures: evidence, elapsedMilliseconds: 1, maximumTensorElements: 16))
        }
        XCTAssertEqual(try grade(LearningExamples.reluReference.apply).status, .passed)
        XCTAssertEqual(try grade { try LearningExamples.reluReference.apply(LearningExamples.reluReference.apply($0)) }.status, .passed)
        let wrong: [(Double) throws -> Double] = [{ $0 }, { Foundation.tanh($0) }, { try LearningExamples.sigmoidReference.apply($0) }, { _ in 0.5 }]
        for transform in wrong {
            XCTAssertEqual(try grade(transform).status, .failed)
        }
    }
    func testRequiredReLUMustReachOutputAndHasNoInventedConfiguration() throws {
        func graph(_ active: Bool = true, config: [String:JSONValue] = [:]) -> GraphDocument {
            GraphDocument(nodes: [.init(id: "i", type: "input", config: ["features": 2]), .init(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2, "bias": false]), .init(id: "s", type: "relu", config: config), .init(id: "o", type: "output")], edges: [.init(source: "i", target: "m"), .init(source: "m", target: active ? "s" : "o")] + (active ? [.init(source: "s", target: "o")] : []))
        }
        XCTAssertNoThrow(try ModelConstructionLessonTasks.validateConstruction(graph(), identity: .relu))
        XCTAssertThrowsError(try ModelConstructionLessonTasks.validateConstruction(graph(false), identity: .relu))
        XCTAssertThrowsError(try ModelConstructionLessonTasks.validateConstruction(graph(config: ["temperature": 2]), identity: .relu))
    }
}
