import XCTest
@testable import KwipMLCore

final class InitializationLearningTests:XCTestCase {
    private func registries()throws->(ComponentRegistry,DataFlowOperatorRegistry) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try (.builtIn(),.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testTwoTasksUseAllFourAlgorithmsAndApproachableRealControls()throws {
        XCTAssertEqual(Set(InitializationLearningTask.allCases.map(\.rawValue)),["initialization-starting-values","initialization-starting-spread"])
        var algorithms=Set<String>()
        for task in InitializationLearningTask.allCases {
            let c=try InitializationLearningConfiguration(values:task.configuration)
            XCTAssertEqual(c.production.seed,42)
            algorithms.formUnion(c.production.rules.map(\.algorithm))
            XCTAssertTrue(task.instructions.contains("Check construction"))
            XCTAssertTrue(task.instructions.contains("seed"));XCTAssertTrue(task.instructions.contains("learned"))
            XCTAssertFalse(task.instructions.contains("derive"))
        }
        XCTAssertEqual(algorithms,["normal","zeros","xavier_uniform","kaiming_uniform"])
    }
    func testFourVariedFixturesDeclareFiveMeasuredZeroDifferences()throws {
        for task in InitializationLearningTask.allCases {
            let plan=try InitializationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:812)
            try plan.validate();XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),["startingSettings","predictions","firstAdjustment","repeatIsolation","savedReuse"])
            for (fixture,w) in zip(plan.fixtures,InitializationLearningTask.widths) {
                let p=w[1]*w[0]+w[1]+w[2]*w[1]+w[2]
                XCTAssertEqual(fixture.inputs["x"]?.shape,[6,w[0]])
                XCTAssertEqual(fixture.inputs["target"]?.shape,[4,w[2]])
                XCTAssertEqual(fixture.expected["startingSettings"]?.shape,[p])
                XCTAssertEqual(fixture.expected["predictions"]?.shape,[6,w[2]])
                XCTAssertEqual(fixture.expected["firstAdjustment"]?.shape,[2,p])
                XCTAssertEqual(fixture.expected["repeatIsolation"]?.shape,[2*p+1])
                XCTAssertEqual(fixture.expected["savedReuse"]?.shape,[p+6*w[2]+1])
                XCTAssertTrue(fixture.expected.values.allSatisfy {$0.values.allSatisfy {$0 == 0}})
            }
            XCTAssertEqual(plan,try InitializationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:812))
            XCTAssertNotEqual(plan.fixtureDigest,try InitializationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:813).fixtureDigest)
        }
    }
    func testValidWrongEvidenceActuallyFailsIndependentCriteria()throws {
        let plan=try InitializationAssessment.plan(task:.startingValues,runtimeVersion:"2.14.1",seed:21)
        let bytes=try InitializationLearningSubmission(graph:InitializationLearningExamples.preparedGraph(task:.startingValues),task:.startingValues).encoded()
        var rows=plan.fixtures.map {LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)}
        let good=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:rows,elapsedMilliseconds:1,maximumTensorElements:1000)
        XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:good).status,.passed)
        for key in ["startingSettings","predictions","firstAdjustment","repeatIsolation","savedReuse"] {
            rows=plan.fixtures.map {LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)}
            rows[0].tensors[key]!.values[0]=0.1
            var bad=good;bad.fixtures=rows
            XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:bad).status,.failed,key)
        }
    }
    func testStrictSubmissionRetainsPartialDraftsAndExecutableWrongControls()throws {
        for task in InitializationLearningTask.allCases {
            let graph=try InitializationLearningExamples.preparedGraph(task:task)
            let bytes=try InitializationLearningSubmission(graph:graph,task:task).encoded()
            XCTAssertEqual(try InitializationLearningSubmission.decode(bytes,task:task).graph,graph)
            var partial=graph;partial.edges.removeAll()
            XCTAssertNoThrow(try InitializationLearningSubmission.validatedGraph(partial,task:task,complete:false))
            XCTAssertThrowsError(try InitializationLearningSubmission(graph:partial,task:task))
            var wrong=graph
            let i=try XCTUnwrap(wrong.nodes.firstIndex {$0.type == "initialization"})
            wrong.nodes[i].config["seed"]=43
            XCTAssertNoThrow(try InitializationLearningSubmission(graph:wrong,task:task))
            var changedOptimizer=graph
            let o=try XCTUnwrap(changedOptimizer.nodes.firstIndex {$0.type == "optimizer_sgd"})
            changedOptimizer.nodes[o].config["lr"]=0.1
            XCTAssertThrowsError(try InitializationLearningSubmission(graph:changedOptimizer,task:task))
            var extra=try JSONDecoder().decode(JSONValue.self,from:bytes)
            if case .object(var object)=extra {object["answers"] = .object([:]);extra = .object(object)}
            XCTAssertThrowsError(try InitializationLearningSubmission.decode(JSONEncoder().encode(extra),task:task))
            XCTAssertThrowsError(try InitializationLearningSubmission.decode(bytes,task:task == .startingValues ? .startingSpread:.startingValues))
        }
    }
    func testReferenceRejectsWrongRolesShapesAndKeepsPrivateTaskAnswerFree()throws {
        let plan=try InitializationAssessment.plan(task:.startingSpread,runtimeVersion:"2.14.1",seed:41)
        var inputs=plan.fixtures[0].inputs
        inputs["unknown"] = .init(shape:[1],values:[0])
        XCTAssertThrowsError(try InitializationLessonReference(inputs:inputs))
        inputs=plan.fixtures[0].inputs;inputs["target"] = .init(shape:[6,2],values:Array(repeating:0,count:12))
        XCTAssertThrowsError(try InitializationLessonReference(inputs:inputs))
        for mode in [LearningMode.guided,.practice,.assessment] {
            let publicTask=plan.publicTask(mode:mode)
            let encoded=String(decoding:try JSONEncoder().encode(publicTask),as:UTF8.self)
            XCTAssertFalse(encoded.contains("expected"))
        }
    }
    func testPreparedExamplesAreActualConnectedWorkflowRecipes()throws {
        let (model,data)=try registries()
        for task in InitializationLearningTask.allCases {
            let document=try (task == .startingValues ? WorkflowExamples.preparedInitializationStartingValues(device:"cpu"):WorkflowExamples.preparedInitializationStartingSpread(device:"cpu"))
            XCTAssertEqual(document.schemaVersion,15)
            let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:13)
            XCTAssertEqual(plan.initializationRecipe?.configuration.values,try InitializationLearningConfiguration(values:task.configuration).values)
            XCTAssertNotNil(plan.model.graph.nodes.first {$0.id == "first"})
            XCTAssertNotNil(plan.model.graph.nodes.first {$0.id == "second"})
            XCTAssertNotNil(plan.optimizerRecipe)
            XCTAssertEqual(plan.optimizerRecipe?.configuration.learningRate,0.05)
        }
    }
}
