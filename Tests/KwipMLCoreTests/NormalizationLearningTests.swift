import XCTest
import Darwin
@testable import KwipMLCore

final class NormalizationLearningTests: XCTestCase {
    func testVariedPlanPopulationOracleAndHistoricalReplay() throws {
        let plan = try NormalizationLessonTask.plan(runtimeVersion: "test", seed: 42)
        XCTAssertEqual(plan.assessmentVersion, ContractVersion(1,2,0))
        XCTAssertEqual(plan.fixtures.count, 4)
        XCTAssertGreaterThan(Set(plan.fixtures.map { $0.expected["scale"]!.values }).count, 2)
        XCTAssertTrue(plan.fixtures.contains { $0.expected["scale"]!.values.contains(1) })
        for fixture in plan.fixtures {
            let rows=fixture.inputs["training"]!.values
            for c in 0..<2 {
                let values=(0..<3).map { rows[$0*2+c] }
                let mean=values.reduce(0,+)/3
                let deviation=sqrt(values.map { ($0-mean)*($0-mean) }.reduce(0,+)/3)
                let scale=deviation==0 ? 1:deviation
                XCTAssertEqual(fixture.expected["mean"]!.values[c],mean,accuracy:1e-12)
                XCTAssertEqual(fixture.expected["scale"]!.values[c],scale,accuracy:1e-12)
                for r in 0..<3 { XCTAssertEqual(try XCTUnwrap(fixture.expected["training"]).values[r*2+c],(values[r]-mean)/scale,accuracy:1e-12) }
                XCTAssertEqual(fixture.expected["validation"]!.values[c],(fixture.inputs["validation"]!.values[c]-mean)/scale,accuracy:1e-12)
            }
        }
        XCTAssertEqual(try NormalizationLessonTask.plan(runtimeVersion:"test",seed:42,assessmentVersion:ContractVersion()), LearningExamples.trainingOnlyStandardization(runtimeVersion:"test",seed:42))
        XCTAssertThrowsError(try NormalizationLessonTask.plan(runtimeVersion:"test",seed:42,assessmentVersion:ContractVersion(1,3)))
        let publicJSON=String(decoding:try JSONEncoder().encode(plan.publicTask(mode:.assessment)),as:UTF8.self)
        XCTAssertFalse(publicJSON.contains("expected"));XCTAssertFalse(publicJSON.contains("fixtureSeed"))
    }
}

extension NormalizationLearningTests {
    private func submission() -> NormalizationLearningSubmission {
        let specs:[(String,String,[String:JSONValue])] = [("source","sourceCSV",[:]),("cast","cast",["columns":.object(["x":.string("float64"),"y":.string("float64")])]),("split","split",["seed":.integer(42),"validationPercent":.integer(25)]),("fit","standardScalerFit",["columns":.array([.string("x"),.string("y")])]),("heldout","transform",[:]),("train","transform",[:])]
        let connections=[("source","table","cast","table"),("cast","table","split","table"),("split","training","fit","training"),("split","validation","heldout","table"),("fit","state","heldout","state"),("split","training","train","table"),("fit","state","train","state")]
        return NormalizationLearningSubmission(graph:DataFlowDocument(nodes:specs.map { DataFlowNode(id:$0.0,operatorID:"kwip.data."+$0.1,config:$0.2) },edges:connections.map { DataFlowEdge(from:.init(node:$0.0,port:$0.1),to:.init(node:$0.2,port:$0.3)) }),outputs:["state":.init(node:"fit",port:"state"),"training":.init(node:"train",port:"table"),"validation":.init(node:"heldout",port:"table")])
    }
    private func receipt(revision:ContractVersion = NormalizationLessonTask.currentVersion) throws -> NormalizationLearningReceipt {
        let plan=try NormalizationLessonTask.plan(runtimeVersion:"test",seed:42,assessmentVersion:revision)
        let value=submission(),hash=learningHash(try NormalizationLearningCodec.encodeSubmission(value))
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:hash,fixtures:plan.fixtures.map { LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected,partitions:["training":["a","b","c"],"fit":["a","b","c"],"validation":["d"]]) },elapsedMilliseconds:10,maximumTensorElements:100)
        let grade=try LearningAssessor.evaluate(plan,evidence:evidence)
        let attempt=LearningAttempt(objectiveID:plan.objectiveID,contentVersion:plan.contentVersion,assessmentVersion:plan.assessmentVersion,mode:.practice,grade:grade)
        return NormalizationLearningReceipt(attempt:attempt,projectID:UUID().uuidString,submission:value,submissionSHA256:hash,runtime:plan.runtime,fixtureSeed:42,fixtureDigest:plan.fixtureDigest,evidence:evidence)
    }
    func testSubmissionStrictAndSemanticIdentityExcludesArrayOrder() throws {
        var value=submission();let first=try value.semanticDigest();value.graph.nodes.reverse();value.graph.edges.reverse()
        XCTAssertEqual(first,try value.semanticDigest())
        value.outputs["validation"] = .init(node:"split",port:"validation")
        XCTAssertNotEqual(first,try value.semanticDigest())
        value=submission();value.graph.nodes[0].config["path"] = .string("private.csv")
        XCTAssertThrowsError(try value.validate())
        value=submission();value.outputs["extra"] = .init(node:"fit",port:"state")
        XCTAssertThrowsError(try value.validate())
        let raw=try NormalizationLearningCodec.encodeSubmission(submission())
        var object=try JSONSerialization.jsonObject(with:raw) as! [String:Any];object["authoredEvidence"] = [:]
        XCTAssertThrowsError(try NormalizationLearningCodec.decodeSubmission(JSONSerialization.data(withJSONObject:object)))
    }
    func testReceiptReplayBindsProjectSubmissionRuntimeFixtureGradeAndPrivacy() throws {
        let original=try receipt()
        XCTAssertNoThrow(try original.verifiedReplay(expectedProjectID:original.projectID))
        XCTAssertEqual(try NormalizationLearningCodec.decodeReceipt(NormalizationLearningCodec.encodeReceipt(original),expectedProjectID:original.projectID).submissionSHA256,original.submissionSHA256)
        XCTAssertThrowsError(try NormalizationLearningCodec.decodeReceipt(NormalizationLearningCodec.encodeReceipt(original),expectedProjectID:UUID().uuidString))
        XCTAssertNotNil(original.formativeProjection(expectedProjectID:original.projectID,currentSubmission:original.submission,currentMode:.practice))
        XCTAssertNil(original.formativeProjection(expectedProjectID:original.projectID,currentSubmission:original.submission,currentMode:.assessment))
        var recorded=original;recorded.attempt.mode = .assessment
        XCTAssertNil(recorded.formativeProjection(expectedProjectID:recorded.projectID,currentSubmission:recorded.submission,currentMode:.practice))
        XCTAssertThrowsError(try original.verifiedReplay(expectedProjectID:UUID().uuidString))
        var changed=original;changed.fixtureSeed += 1;XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:changed.projectID))
        changed=original;changed.runtime.version="different";XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:changed.projectID))
        changed=original;changed.attempt.objectiveID="other";XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:changed.projectID))
        changed=original;changed.attempt.grade.summary="authored";XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:changed.projectID))
        var current=original.submission;current.outputs["validation"] = .init(node:"split",port:"validation")
        XCTAssertNil(original.formativeProjection(expectedProjectID:original.projectID,currentSubmission:current,currentMode:.practice))
        changed=original;changed.submission=current;XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:changed.projectID))
    }
    func testHistoricalReceiptProjectionRetainsExactVersionedEvidenceKeys() throws {
        for revision in [ContractVersion(),ContractVersion(1,1),ContractVersion(1,2)] {
            let value=try receipt(revision:revision)
            let restored=try NormalizationLearningCodec.decodeReceipt(NormalizationLearningCodec.encodeReceipt(value),expectedProjectID:value.projectID)
            let projected=try XCTUnwrap(restored.formativeProjection(expectedProjectID:value.projectID,currentSubmission:value.submission,currentMode:.guided))
            XCTAssertEqual(Set(projected.fixtures[0].measured.keys),revision == ContractVersion(1,2) ? ["mean","scale","training","validation"]:["mean","scale","validation"])
            XCTAssertNil(restored.formativeProjection(expectedProjectID:value.projectID,currentSubmission:value.submission,currentMode:.assessment))
            var assessment=restored;assessment.attempt.mode = .assessment
            XCTAssertNil(assessment.formativeProjection(expectedProjectID:value.projectID,currentSubmission:value.submission,currentMode:.guided))
        }
    }
    func testDraftBoundsAndRoundTrip() throws {
        var draft=NormalizationLearningDraft(projectID:UUID().uuidString,submission:submission())
        XCTAssertEqual(try NormalizationLearningCodec.decodeDraft(NormalizationLearningCodec.encodeDraft(draft)),draft)
        draft.projectID="invalid";XCTAssertThrowsError(try draft.validate())
        draft.projectID=UUID().uuidString;draft.notes=String(repeating:"x",count:8001);XCTAssertThrowsError(try draft.validate())
    }
}

extension NormalizationLearningTests {
    func testCurrentWorkerRefusesRetiredConstructionWithoutMintingHistoricalEvidence() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else { throw XCTSkip("Development Python unavailable") }
        func invoke(_ args:[String]) throws -> Data {
            let process=Process();let pipe=Pipe();process.executableURL=python;process.arguments=[root.appendingPathComponent("runner/main.py").path]+args;process.standardOutput=pipe
            try process.run();let data=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0,String(decoding:data,as:UTF8.self));return data
        }
        struct RuntimeEvent:Decodable { var versions:[String:String] }
        let version=try XCTUnwrap(JSONDecoder().decode(RuntimeEvent.self,from:invoke(["--learning-runtime"])).versions["numpy"])
        let temporary=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:true)
        // The worker anchors its managed fixture owner without following parent symlinks.
        let resolved=try XCTUnwrap(realpath(temporary.path,nil))
        let physicalDirectory=String(cString:resolved);free(resolved)
        let directory=URL(fileURLWithPath:physicalDirectory)
        defer { try? FileManager.default.removeItem(at:directory) }
        struct Fixture:Encodable { var id:String;var inputs:[String:LearningTensor] }
        struct Request:Encodable { var assessmentID:String;var assessmentVersion:ContractVersion;var fixtureDigest:String;var runtime:LearningRuntime;var budget:LearningBudget;var fixtures:[Fixture] }
        struct Event:Decodable { var evidence:LearningEvidence }
        for revision in [ContractVersion(1,1),ContractVersion(1,2)] {
            let plan=try NormalizationLessonTask.plan(runtimeVersion:version,seed:42,assessmentVersion:revision)
            let request=Request(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,budget:plan.budget,fixtures:plan.fixtures.map { Fixture(id:$0.id,inputs:$0.inputs) })
            let path=directory.appendingPathComponent("request.json"), submitted=directory.appendingPathComponent("submission.json")
            try JSONEncoder().encode(request).write(to:path)
            let submittedGraph=submission()
            let data=try NormalizationLearningCodec.encodeSubmission(submittedGraph);try data.write(to:submitted)
            let evidence=try JSONDecoder().decode(Event.self,from:invoke(["--learning-evidence-request",physicalDirectory+"/request.json","--learning-submission",physicalDirectory+"/submission.json","--learning-submission-sha256",learningHash(data)])).evidence
            XCTAssertTrue(evidence.runtime.capabilities.isEmpty,"Current worker cannot advertise retired scaler/transform capabilities")
            XCTAssertEqual(evidence.maximumTensorElements,0,"Capability refusal happens before numerical allocation")
            XCTAssertTrue(evidence.fixtures.isEmpty,"Refusal cannot fabricate historical numerical proof")
            let grade=try LearningAssessor.evaluate(plan,evidence:evidence)
            XCTAssertEqual(grade.status,.unavailable)
        }
    }
}

extension NormalizationLearningTests {
    func testIncompleteDraftPersistsButCannotExecute() throws {
        let empty=NormalizationLearningSubmission(graph:DataFlowDocument(),outputs:[:])
        var draft=NormalizationLearningDraft(projectID:UUID().uuidString,submission:empty)
        XCTAssertEqual(try NormalizationLearningCodec.decodeDraft(NormalizationLearningCodec.encodeDraft(draft)),draft)
        XCTAssertThrowsError(try draft.submission.validate())
        draft.submission=submission();draft.submission.outputs["state"] = .init(node:"deleted-node",port:"state")
        XCTAssertEqual(try NormalizationLearningCodec.decodeDraft(NormalizationLearningCodec.encodeDraft(draft)),draft)
        XCTAssertThrowsError(try draft.submission.validate())
        draft.returnProjectID=draft.projectID;XCTAssertThrowsError(try draft.validate())
    }
}
