import XCTest
@testable import KwipMLCore
final class AttentionLearningTests: XCTestCase {
    static func graph(_ task: AttentionLearningTask) -> GraphDocument {
        var nodes = task.sourceRoles.map { GraphNode(id:$0,type:"input",config:["fixture":.string($0)]) }
        nodes += [.init(id:"mask",type:"attention_mask",config:task.maskConfiguration),.init(id:"op",type:task.componentID,config:task.configuration)]
        nodes += task.outputRoles.map { .init(id:"output-"+$0,type:"output",config:["role":.string($0)]) }
        var edges = task.sourceRoles.map { GraphEdge(id:"input-"+$0,source:$0,sourcePort:"out",target:"op",targetPort:$0) }
        for (i,role) in task.maskReferenceRoles.enumerated() { edges.append(.init(id:"reference-\(i)",source:role,sourcePort:"out",target:"mask",targetPort:i == 0 ? "query":"key")) }
        edges.append(.init(id:"blocked",source:"mask",sourcePort:"mask",target:"op",targetPort:task.maskInputPort))
        if task == .multihead { edges.append(.init(id:"padding",source:"mask",sourcePort:"padding",target:"op",targetPort:"key_padding_mask")) }
        edges += task.outputRoles.map { .init(id:"output-"+$0,source:"op",sourcePort:$0 == "result" ? "out":$0,target:"output-"+$0,targetPort:"in") }
        return .init(name:"Compare useful positions",nodes:nodes,edges:edges)
    }
    func testIndependentHandCalculatedScaledMixtureAndInfluence() throws {
        let task = AttentionLearningTask.scaled
        let inputs: [String: LearningTensor] = ["query":.init(shape:[1,1,2,2],values:[0,0,0,0]),"key":.init(shape:[1,1,3,2],values:Array(repeating:0,count:6)),"value":.init(shape:[1,1,3,2],values:[1,2,7,8,5,6]),"upstream.result":.init(shape:[1,1,2,2],values:[1,1,1,1])]
        let gold = try AttentionLessonReference(inputs:inputs,task:task).expected
        let expected = [3.0,4.0,13.0/3,16.0/3]
        for (a,b) in zip(gold["result"]!.values,expected) { XCTAssertEqual(a,b,accuracy:1e-12) }
        for (a,b) in zip(gold["valueGradient"]!.values,[5.0/6,5.0/6,1.0/3,1.0/3,5.0/6,5.0/6]) { XCTAssertEqual(a,b,accuracy:2e-9) }
        XCTAssertFalse(gold.keys.contains { $0.hasPrefix("parameter.") || $0.hasPrefix("mask") })
    }
    func testFiveFourFixturePlansAndNativeOwnership() throws {
        for task in AttentionLearningTask.allCases {
            let plan = try AttentionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173)
            XCTAssertEqual(plan.fixtures.count,4); XCTAssertEqual(Set(plan.fixtures.map(\.id)).count,4)
            XCTAssertEqual(plan.runtime.capabilities.keys.sorted(),[task.componentID,"attention_mask"].sorted())
            let c = try AttentionConfiguration(componentID:task.componentID,values:task.configuration)
            XCTAssertEqual(Dictionary(uniqueKeysWithValues:c.parameterShapes.map { ($0.name,$0.shape) }),task.parameterShapes)
            for f in plan.fixtures { XCTAssertTrue(f.expected.values.allSatisfy(\.wellFormed)); for role in task.sourceRoles { XCTAssertEqual(f.expected[role+"Gradient"]?.shape,f.inputs[role]?.shape) }; for key in task.parameterShapes.keys { XCTAssertEqual(f.expected["parameter."+key+"Gradient"]?.shape,task.parameterShapes[key]) } }
            XCTAssertTrue(task.instructions.contains("Build:")); XCTAssertTrue(task.instructions.contains("Configure:")); XCTAssertTrue(task.instructions.contains("Check:")); XCTAssertTrue(task.explanation.contains("query")); XCTAssertTrue(task.explanation.contains("1 means blocked"))
        }
    }
    func testStrictMaskUtilityWiringAndPartialGeometry() throws {
        for task in AttentionLearningTask.allCases {
            let g = Self.graph(task), submission = try AttentionLearningSubmission(graph:g,task:task)
            XCTAssertEqual(try AttentionLearningSubmission.decode(submission.encoded(),task:task),submission)
            XCTAssertTrue(AttentionLearningCanvas.inspect(g,task:task).issues.isEmpty)
            let inspection = AttentionLearningCanvas.inspect(g,task:task)
            XCTAssertEqual(inspection.outputTensorContracts[.init("mask","mask")]?.dtype,.int64)
            for role in task.outputRoles { XCTAssertEqual(inspection.outputTensorContracts[.init("op",role == "result" ? "out":role)]?.layout.sample.dimensions,task.outputSampleShapes[role]) }
            var missing = g; missing.edges.removeLast(); XCTAssertNoThrow(try AttentionLearningSubmission.validatedGraph(missing,task:task,complete:false)); XCTAssertThrowsError(try AttentionLearningSubmission(graph:missing,task:task))
            var wrong = g; wrong.edges[0].targetPort = "not-a-role"; XCTAssertThrowsError(try AttentionLearningSubmission.validatedGraph(wrong,task:task,complete:false))
            var cheating = g; cheating.nodes[0].config["answer"] = .number(1); XCTAssertThrowsError(try AttentionLearningSubmission.validatedGraph(cheating,task:task,complete:false))
        }
    }
}

extension AttentionLearningTests {
    func testAttentionTransportExactKeysShapesAndCriteriaWhilePriorBoundsRemain() throws {
        var old = try TensorLearningTask.plan(operation:"reshape",runtimeVersion:"2.14.1",seed:41)
        old.fixtures[0].inputs = Dictionary(uniqueKeysWithValues:(0..<17).map { ("old\($0)",LearningTensor(shape:[1],values:[1])) })
        XCTAssertThrowsError(try old.validate())
        old = try TensorLearningTask.plan(operation:"reshape",runtimeVersion:"2.14.1",seed:41)
        let criterion = old.criteria[0]
        old.criteria = (0..<33).map { .init(id:"old\($0)",description:criterion.description,nextStep:criterion.nextStep,rule:criterion.rule) }
        XCTAssertThrowsError(try old.validate())
        let valid = try AttentionAssessment.plan(task:.transformer,runtimeVersion:"2.14.1",seed:41)
        XCTAssertEqual(valid.fixtures[0].inputs.count,37); XCTAssertEqual(valid.criteria.count,39); XCTAssertNoThrow(try valid.validate())
        for mutation in 0..<6 {
            var plan = valid
            switch mutation {
            case 0: plan.fixtures[0].inputs["answer"] = .init(shape:[1],values:[1])
            case 1: plan.fixtures[0].expected.removeValue(forKey:"sourceGradient")
            case 2: plan.fixtures[0].inputs["source"] = .init(shape:[1,2,3],values:Array(repeating:1,count:6))
            case 3: plan.runtime.capabilities.removeValue(forKey:"attention_mask")
            case 4: plan.criteria.removeLast()
            default: plan.criteria[0].rule = .tensor(key:"result",absoluteTolerance:0.5,relativeTolerance:0.5)
            }
            XCTAssertThrowsError(try plan.validate(),"mutation \(mutation)")
        }
    }
}

extension AttentionLearningTests {
    func testPrivateTransformerSeedReferenceConvergesToIndependentNativeInfluence() throws {
        // Captured actual Core-generated fixture and independent native float64
        // derivatives: .local/attention-core-seed-diagnostics/plan-235.json.
        // The unchanged scalar reference must remove differentiation truncation;
        // loosening the grader's tolerance would conceal this teaching error.
        let plan=try AttentionAssessment.plan(task:.transformer,runtimeVersion:"2.14.1",seed:5771968745342428653)
        let influence=try XCTUnwrap(plan.fixtures[1].expected["sourceGradient"])
        let native=[-65.94130353488612,65.94130353488612,-75.03606275136744,75.03606275136741]
        for (offset,value) in native.enumerated() {XCTAssertEqual(influence.values[6+offset],value,accuracy:1e-6)}
        for criterion in plan.criteria {
            guard case .tensor(_,let absolute,let relative)=criterion.rule else {return XCTFail("Require numerical influence criteria.")}
            XCTAssertEqual(absolute,["blockedConnections","blockedSourcePositions"].contains(criterion.id) ? 0:3e-6);XCTAssertEqual(relative,["blockedConnections","blockedSourcePositions"].contains(criterion.id) ? 0:3e-5)
        }
    }
}

extension AttentionLearningTests {
    func testMaskCheckpointsRequireExactBinaryGeometryAndFreshReceiptEvidence() throws {
        for task in AttentionLearningTask.allCases {
            let plan=try AttentionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41)
            let q=task.sampleShapes[task.maskReferenceRoles[0]]!.suffix(2).first!,s=task.sampleShapes[task.maskReferenceRoles[1]]!.suffix(2).first!
            XCTAssertEqual(plan.fixtures[0].expected["blockedConnections"],LearningTensor(shape:[1,q,s],values:(0..<q).flatMap {i in (0..<s).map {j in i==0 && j==1 ? 1.0:0.0}}))
            XCTAssertEqual(plan.fixtures[0].expected["blockedSourcePositions"],LearningTensor(shape:[1,s],values:Array(repeating:0,count:s)))
            XCTAssertTrue(plan.criteria.contains {$0.id == "blockedConnections"});XCTAssertTrue(plan.criteria.contains {$0.id == "blockedSourcePositions"})
            let data=try AttentionLearningSubmission(graph:Self.graph(task),task:task).encoded()
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(data),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:36)
            let receipt=try TensorLearningReceipt(projectID:"D892EB29-7D4D-4B1D-BC59-A6A592D652E3",operation:task.rawValue,mode:.guided,submissionData:data,plan:plan,evidence:evidence)
            XCTAssertNotNil(receipt.attentionFormativeProjection(expectedProjectID:"D892EB29-7D4D-4B1D-BC59-A6A592D652E3",currentSubmission:data,currentMode:.guided))
            XCTAssertNil(receipt.attentionFormativeProjection(expectedProjectID:"D892EB29-7D4D-4B1D-BC59-A6A592D652E3",currentSubmission:data,currentMode:.assessment))
            var missing=receipt;missing.evidence.fixtures[0].tensors.removeValue(forKey:"blockedConnections")
            XCTAssertThrowsError(try missing.encoded())
            XCTAssertNil(missing.attentionFormativeProjection(expectedProjectID:"D892EB29-7D4D-4B1D-BC59-A6A592D652E3",currentSubmission:data,currentMode:.guided))
            for mutation in 0..<4 {
                var invalid=plan
                switch mutation {
                case 0:invalid.fixtures[0].expected.removeValue(forKey:"blockedSourcePositions")
                case 1:invalid.fixtures[0].expected["blockedConnections"] = .init(shape:[1,s,q],values:Array(repeating:0,count:q*s))
                case 2:invalid.fixtures[0].expected["blockedConnections"] = .init(shape:[1,q,s],values:Array(repeating:0.5,count:q*s))
                default:invalid.criteria.removeAll {$0.id == "blockedConnections"}
                }
                XCTAssertThrowsError(try invalid.validate())
            }
        }
    }
}
