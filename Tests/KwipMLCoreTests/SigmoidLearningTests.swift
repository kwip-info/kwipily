import XCTest
@testable import KwipMLCore

final class SigmoidLearningTests: XCTestCase {
    func testIndependentReferencePublicNumbersSymmetryAndFiniteBoundary() throws {
        let reference = LearningExamples.sigmoidReference
        for (x, expected) in [(-2.0, 0.11920292202211755), (0, 0.5), (1, 0.7310585786300049), (2, 0.8807970779778823)] {
            XCTAssertEqual(try reference.apply(x), expected, accuracy: 1e-14)
            XCTAssertEqual(try reference.apply(-x) + reference.apply(x), 1, accuracy: 1e-14)
        }
        XCTAssertEqual(try reference.apply(-1000), 0)
        XCTAssertEqual(try reference.apply(1000), 1)
        XCTAssertThrowsError(try reference.apply(.nan)); XCTAssertThrowsError(try reference.apply(.infinity))
        XCTAssertTrue(reference.instructions.contains("not automatically calibrated probabilities"))
    }
    func testVersionedTaskSeedReplayAssessmentSecrecyAndMisconceptions() throws {
        let identity = ModelConstructionLessonIdentity.sigmoid
        try identity.validate()
        XCTAssertEqual(try JSONDecoder().decode(ModelConstructionLessonIdentity.self, from: JSONEncoder().encode(identity)), identity)
        let task = try ModelConstructionLessonTasks.publicTask(identity: identity)
        XCTAssertEqual(task.allowedNodeTypes, ["input", "linear", "sigmoid", "add", "concat", "output"])
        let plan = try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 42)
        XCTAssertEqual(plan.fixtureDigest, try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 42).fixtureDigest)
        XCTAssertNotEqual(plan.fixtureDigest, try ModelConstructionLessonTasks.plan(identity: identity, runtimeVersion: "reviewed-test-runtime", fixtureSeed: 43).fixtureDigest)
        XCTAssertEqual(plan.runtime.capabilities, ["linear": ContractVersion(1,1,0), "sigmoid": ContractVersion()])
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
        XCTAssertEqual(try grade(LearningExamples.sigmoidReference.apply).status, .passed)
        let wrong: [(Double) throws -> Double] = [{ $0 }, { max(0,$0) }, { tanh($0) }, { _ in 0.5 }, { try LearningExamples.sigmoidReference.apply(LearningExamples.sigmoidReference.apply($0)) }]
        for transform in wrong {
            XCTAssertEqual(try grade(transform).status, .failed)
        }
    }
    func testRequiredSigmoidMustReachOutputAndHasNoInventedConfiguration() throws {
        func graph(_ active: Bool = true, config: [String:JSONValue] = [:]) -> GraphDocument {
            GraphDocument(nodes: [.init(id: "i", type: "input", config: ["features": 2]), .init(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2, "bias": false]), .init(id: "s", type: "sigmoid", config: config), .init(id: "o", type: "output")], edges: [.init(source: "i", target: "m"), .init(source: "m", target: active ? "s" : "o")] + (active ? [.init(source: "s", target: "o")] : []))
        }
        XCTAssertNoThrow(try ModelConstructionLessonTasks.validateConstruction(graph(), identity: .sigmoid))
        XCTAssertThrowsError(try ModelConstructionLessonTasks.validateConstruction(graph(false), identity: .sigmoid))
        XCTAssertThrowsError(try ModelConstructionLessonTasks.validateConstruction(graph(config: ["temperature": 2]), identity: .sigmoid))
    }
}
