import Foundation
import Darwin
import XCTest
@testable import KwipMLCore

final class EmbeddingBagWorkerBridgeTests:XCTestCase {
    func testEveryPrivateSwiftPlanMatchesPinnedDispatcherAndReplaysActualReceipt() throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:directory)}
        let plans=try EmbeddingBagLearningTask.allCases.map {try EmbeddingBagAssessment.plan(task:$0,runtimeVersion:"numerical-bridge",seed:31)}
        var requests:[JSONValue]=[]
        for (task,plan) in zip(EmbeddingBagLearningTask.allCases,plans) {
            let submission=try EmbeddingBagLearningSubmission(graph:EmbeddingBagExamples.lessonGraph(task:task))
            let fixtures=try plan.fixtures.map {fixture -> JSONValue in
                .object(["id":.string(fixture.id),"inputs":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(fixture.inputs))])
            }
            func wire<T:Encodable>(_ value:T)throws->JSONValue {try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(value))}
            requests.append(.object(["submission":try wire(submission),"request":.object([
                "assessmentID":.string(plan.id),"assessmentVersion":try wire(plan.assessmentVersion),
                "fixtureDigest":.string(plan.fixtureDigest),"runtime":try wire(plan.runtime),
                "budget":try wire(plan.budget),"fixtures":.array(fixtures)])]))
        }
        let input=directory.appendingPathComponent("input.json"),output=directory.appendingPathComponent("output.json"),errors=directory.appendingPathComponent("errors.txt")
        try JSONEncoder().encode(requests).write(to:input)
        FileManager.default.createFile(atPath:output.path,contents:nil);FileManager.default.createFile(atPath:errors.path,contents:nil)
        let process=Process();process.executableURL=repo.appendingPathComponent(".venv/bin/python");process.currentDirectoryURL=repo
        process.arguments=["-B","-c",#"""
import json,sys,torch,tempfile,pathlib,hashlib
from runner.learning_evidence import execute_request
results=[]
for item in json.load(sys.stdin):
    request=item['request'];request['runtime']['version']=str(torch.__version__)
    with tempfile.TemporaryDirectory() as folder:
        root=pathlib.Path(folder)
        raw=json.dumps(item['submission'],sort_keys=True,separators=(',',':')).encode()
        (root/'request.json').write_text(json.dumps(request))
        (root/'submission.json').write_bytes(raw)
        evidence=execute_request(root/'request.json',root/'submission.json',hashlib.sha256(raw).hexdigest())
        results.append(evidence)
json.dump({'torchVersion':str(torch.__version__),'results':results},sys.stdout,allow_nan=False)
"""#]
        let stdin=try FileHandle(forReadingFrom:input),stdout=try FileHandle(forWritingTo:output),stderr=try FileHandle(forWritingTo:errors)
        defer {try? stdin.close();try? stdout.close();try? stderr.close()}
        process.standardInput=stdin;process.standardOutput=stdout;process.standardError=stderr
        try process.run();let deadline=Date().addingTimeInterval(40)
        while process.isRunning && Date()<deadline {Thread.sleep(forTimeInterval:0.025)}
        if process.isRunning {_ = Darwin.kill(process.processIdentifier,SIGKILL)};process.waitUntilExit()
        let diagnostic=String(decoding:try Data(contentsOf:errors),as:UTF8.self)
        XCTAssertEqual(process.terminationStatus,0,diagnostic)
        let raw=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:output))
        XCTAssertFalse(try XCTUnwrap(raw["torchVersion"]?.stringValue).isEmpty)
        guard case .array(let results)=raw["results"] else {return XCTFail("Missing actual native measurements")}
        XCTAssertEqual(results.count,plans.count)
        for ((task,originalPlan),result) in zip(zip(EmbeddingBagLearningTask.allCases,plans),results) {
            var plan=originalPlan;plan.runtime.version=try XCTUnwrap(raw["torchVersion"]?.stringValue)
            let evidence=try JSONDecoder().decode(LearningEvidence.self,from:JSONEncoder().encode(result))
            XCTAssertNil(evidence.backendError)
            XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.passed)
            let submission=try EmbeddingBagLearningSubmission(graph:EmbeddingBagExamples.lessonGraph(task:task)).encoded()
            let receipt=try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice,submissionData:submission,plan:plan,evidence:evidence)
            let restored=try TensorLearningReceipt.decode(receipt.encoded())
            XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:receipt.projectID),plan)
            let fixtures=evidence.fixtures
            XCTAssertEqual(fixtures.map(\.fixtureID),plan.fixtures.map(\.id))
            for (expected,actual) in zip(plan.fixtures,fixtures) {
                XCTAssertEqual(Set(actual.tensors.keys),Set(expected.expected.keys))
                for (name,tensor) in expected.expected {
                    let measured=try XCTUnwrap(actual.tensors[name]);XCTAssertEqual(measured.shape,tensor.shape)
                    XCTAssertEqual(measured.values.count,tensor.values.count)
                    for (a,b) in zip(measured.values,tensor.values) {XCTAssertEqual(a,b,accuracy:3e-6+3e-5*abs(b),"\(plan.objectiveID): \(name)")}
                }
            }
        }
    }
}
