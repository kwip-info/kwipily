import Foundation
import XCTest
@testable import KwipMLCore

final class ModuleCompositionLearningTests: XCTestCase {
    func testIndependentBranchAndNoncommutingSequenceHandCalculations() throws {
        let coefficients = [2.0,1.0,3.0,-2.0], rows = [-2.0,0.5,3.0]
        XCTAssertEqual(try ModuleCompositionLessonReference.output(coefficients: coefficients, rows: rows, kind: "branch"), [-14,6,26])
        XCTAssertEqual(try ModuleCompositionLessonReference.output(coefficients: coefficients, rows: rows, kind: "sequenceAB"), [-11,4,19])
        XCTAssertEqual(try ModuleCompositionLessonReference.output(coefficients: coefficients, rows: rows, kind: "sequenceBA"), [-15,0,15])
        XCTAssertThrowsError(try ModuleCompositionLessonReference.output(coefficients: coefficients, rows: rows, kind: "unreviewed"))
        XCTAssertThrowsError(try ModuleCompositionLessonReference.output(coefficients: [2,.infinity,3,-2], rows: rows, kind: "branch"))
    }
    func testFourVariedSeededFixturesHaveFiveExactRolesAndColdIdentity() throws {
        for task in ModuleCompositionLearningTask.allCases {
            let plan = try task.assessment(seed: 17, runtimeVersion: "2.14.1")
            XCTAssertEqual(plan.id, "assessment." + task.rawValue)
            XCTAssertEqual(plan.fixtures.count, 4); XCTAssertEqual(plan.criteria.count, 5)
            XCTAssertEqual(Set(plan.criteria.map(\.id)), Set(task.measuredShapes.keys))
            XCTAssertEqual(plan, try task.assessment(seed: 17, runtimeVersion: "2.14.1"))
            XCTAssertNotEqual(plan.fixtures, try task.assessment(seed: 18, runtimeVersion: "2.14.1").fixtures)
            for fixture in plan.fixtures {
                XCTAssertEqual(fixture.expected.mapValues(\.shape), task.measuredShapes)
                XCTAssertTrue(fixture.expected.values.allSatisfy { $0.wellFormed })
                XCTAssertEqual(fixture.inputs["coefficients"]?.shape, [4])
                XCTAssertEqual(fixture.inputs["rows"]?.shape, [3,1])
                XCTAssertEqual(fixture.inputs["upstream"]?.shape, [3,1])
                XCTAssertEqual(Set(fixture.inputs["rows"]!.values.map { Float($0).bitPattern }).count, 3)
            }
        }
    }
    func testPreparedRealSourcesExpandToNativePoliciesAndPrivateInstances() throws {
        let registry = try ComponentRegistry.builtIn()
        for task in ModuleCompositionLearningTask.allCases {
            let source = try ModuleCompositionLearningExamples.preparedSource(task: task)
            let submission = try ModuleCompositionLearningSubmission(source: source, task: task, registry: registry)
            XCTAssertEqual(try ModuleCompositionLearningSubmission.decode(submission.encoded(), task: task, registry: registry), submission)
            let expansion = try CompositeExpander.expand(source, registry: registry, context: .explicitObjective)
            let plan = try XCTUnwrap(expansion.graph.compositionPlan)
            XCTAssertEqual(source.schemaVersion, 3)
            if task == .buildReuse {
                XCTAssertTrue(plan.containers.contains { $0.path.contains("left") })
                XCTAssertTrue(plan.containers.contains { $0.path.contains("right") })
                XCTAssertTrue(expansion.graph.nodes.contains { $0.type == "dropout" })
            } else {
                XCTAssertTrue(Set(plan.containers.map(\.kind)).isSuperset(of: [.sequential,.orderedList,.namedMap]))
            }
        }
    }
    func testIncompleteDraftCanOpenButCannotBecomeCompletedSubmission() throws {
        let registry = try ComponentRegistry.builtIn()
        for task in ModuleCompositionLearningTask.allCases {
            let source = try ModuleCompositionLearningExamples.defaultSource(task: task)
            XCTAssertEqual(try ModuleCompositionLearningSubmission.validatedSource(source, task: task, registry: registry, complete: false), source)
            XCTAssertThrowsError(try ModuleCompositionLearningSubmission(source: source, task: task, registry: registry))
        }
    }
    func testProtectedModelAndEnvelopeRejectAnswerInjectionAndWrongTask() throws {
        let registry = try ComponentRegistry.builtIn(), task = ModuleCompositionLearningTask.buildReuse
        let source = try ModuleCompositionLearningExamples.preparedSource(task: task)
        var changed = source; changed.definitions[0].body.nodes[0].config["out_features"] = 2
        XCTAssertThrowsError(try ModuleCompositionLearningSubmission(source: changed, task: task, registry: registry))
        let submission = try ModuleCompositionLearningSubmission(source: source, task: task, registry: registry)
        XCTAssertThrowsError(try ModuleCompositionLearningSubmission.decode(submission.encoded(), task: .orderName, registry: registry))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: submission.encoded()) as? [String:Any])
        object["answers"] = [0,0,0]
        XCTAssertThrowsError(try ModuleCompositionLearningSubmission.decode(JSONSerialization.data(withJSONObject: object), task: task, registry: registry))
        object.removeValue(forKey: "answers"); object["schemaVersion"] = true
        XCTAssertThrowsError(try ModuleCompositionLearningSubmission.decode(JSONSerialization.data(withJSONObject: object), task: task, registry: registry))
    }
    func testProjectionRequiresAllCurrentCriteriaAndNeverExposesPrivateTables() throws {
        let task=ModuleCompositionLearningTask.orderName,registry=try ComponentRegistry.builtIn()
        let plan=try task.assessment(seed:42,runtimeVersion:"2.14.1")
        let submission=try ModuleCompositionLearningSubmission(source:ModuleCompositionLearningExamples.preparedSource(task:task),task:task,registry:registry).encoded()
        // Synthetic transport fixture tests admission only; it is not a native execution receipt.
        var evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(submission),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:128)
        func project(_ mode:LearningMode = .guided)->ModuleCompositionFormativeProjection? {
            .project(plan:plan,evidence:evidence,currentSubmission:submission,registry:registry,mode:mode,attemptMode:mode)
        }
        XCTAssertNotNil(project());XCTAssertNil(project(.assessment))
        evidence.submissionDigest=String(repeating:"0",count:64);XCTAssertNil(project())
        evidence.submissionDigest=learningHash(submission);evidence.fixtures[0].tensors.removeValue(forKey:"structuralIdentity");XCTAssertNil(project())
        evidence.fixtures[0].tensors=plan.fixtures[0].expected;evidence.fixtureDigest="old";XCTAssertNil(project())
    }

    func testStartingDraftRequiresLearnerInternalConnectionsAndContainerChoices() throws {
        for task in ModuleCompositionLearningTask.allCases {
            let source=try ModuleCompositionLearningExamples.defaultSource(task:task)
            XCTAssertTrue(source.root.edges.isEmpty)
            let rootPositions=source.root.nodes.map { "\($0.x),\($0.y)" } + source.root.instances.map { "\($0.x),\($0.y)" }
            XCTAssertEqual(Set(rootPositions).count,rootPositions.count,"Distinct supplied blocks need distinct initial canvas positions.")
            for definition in source.definitions {
                let positions=definition.body.nodes.map { "\($0.x),\($0.y)" } + definition.body.instances.map { "\($0.x),\($0.y)" }
                XCTAssertEqual(Set(positions).count,positions.count)
            }
            for definition in source.definitions {
                XCTAssertTrue(definition.body.edges.isEmpty,"Starting drafts must leave internal connections for the learner.")
                if task == .orderName {
                    XCTAssertEqual(definition.composition?.kind,.graph)
                    XCTAssertEqual(definition.composition?.children.map(\.localID),["a","b"])
                    XCTAssertTrue(definition.composition?.children.allSatisfy {$0.key == nil} == true)
                }
            }
            let registry=try ComponentRegistry.builtIn()
            XCTAssertNoThrow(try ModuleCompositionLearningSubmission.validatedSource(source,task:task,registry:registry,complete:false))
            XCTAssertThrowsError(try ModuleCompositionLearningSubmission(source:source,task:task,registry:registry))
        }
    }

}
