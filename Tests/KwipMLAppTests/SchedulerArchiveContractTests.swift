import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SchedulerArchiveContractTests: XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func compile(_ document:WorkflowDocument,version:Int)throws->WorkflowExecutionPlan {
        let train=try XCTUnwrap(document.nodes.first {if case .train = $0.payload {return true};return false})
        return try WorkflowCompiler.compile(document,train:train.address,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:version)
    }
    private func object<T:Encodable>(_ value:T)throws->[String:Any] {try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(value)) as? [String:Any])}
    private func capture(_ document:WorkflowDocument,version:Int)throws->WorkflowArchiveCapture {
        let plan=try compile(document,version:version),pin=String(repeating:"a",count:64)
        func file(_ path:String)->WorkflowArchiveFile {.init(path:path,bytes:10,sha256:pin)}
        return CurrentArchiveFixture.current(.init(schemaVersion:document.schemaVersion,compilerContractVersion:version,projectID:document.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,revision:document.revision,train:plan.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),sources:plan.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},plan:try WorkflowArchivePlan(plan,compilerContractVersion:version)))
    }
    func testFiveGenuineCompiledRecipesRoundTripAndDetectRecipeRateOrQualifiedSourceChanges()throws {
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let document=try WorkflowExamples.preparedSchedulerClassifier(componentID:id,device:"cpu")
            let plan=try compile(document,version:12),archive=try WorkflowArchivePlan(plan,compilerContractVersion:12)
            let encodedRecipe=try XCTUnwrap(object(archive)["schedulerRecipe"] as? [String:Any])
            XCTAssertEqual(Set(encodedRecipe.keys),["source","configuration"])
            let expected=try XCTUnwrap(plan.schedulerRecipe)
            XCTAssertEqual(expected.source,plan.train)
            XCTAssertEqual(try JSONDecoder().decode(WorkflowNodeAddress.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(encodedRecipe["source"]))),plan.train)
            let value=try JSONDecoder().decode(JSONValue.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(encodedRecipe["configuration"])))
            XCTAssertEqual(try SchedulerConfiguration.decodeRecipe(value),expected.configuration)
            XCTAssertEqual(Set(try XCTUnwrap(encodedRecipe["configuration"] as? [String:Any]).keys),["schemaVersion","componentID","configurationVersion","config"])
            let restored=try JSONDecoder().decode(WorkflowArchivePlan.self,from:WorkflowArchiveCodec.encode(archive))
            XCTAssertEqual(restored,archive);XCTAssertTrue(try restored.matches(plan,compilerContractVersion:12))
            let captured=try capture(document,version:12)
            XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(captured)),captured)
            let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==plan.train})
            let adjustableKey = ["scheduler_step_lr":"gamma","scheduler_cosine_annealing_lr":"eta_min","scheduler_reduce_on_plateau":"threshold","scheduler_one_cycle_lr":"max_lr","scheduler_linear_lr":"start_factor"][id]!
            for mutation in ["algorithm","rate","address"] {
                var changed=document
                guard case .train(var config)=changed.nodes[index].payload else {throw WorkflowArchiveError("Missing Train")}
                if mutation == "algorithm" {config["scheduler"]=try SchedulerConfiguration(componentID:id == "scheduler_step_lr" ? "scheduler_linear_lr":"scheduler_step_lr",values:[:]).recipeValue}
                if mutation == "rate" {config["scheduler"]=try SchedulerConfiguration(componentID:id,values:[adjustableKey:id == "scheduler_cosine_annealing_lr" ? 0.01:0.07]).recipeValue}
                changed.nodes[index].payload = .train(config:config)
                if mutation == "address" {
                    let old=changed.nodes[index].address,new=WorkflowNodeAddress(domain:.control,localID:"different-train")
                    changed.nodes[index].address=new
                    for edge in changed.edges.indices {
                        if changed.edges[edge].from.address==old {changed.edges[edge].from.address=new}
                        if changed.edges[edge].to.address==old {changed.edges[edge].to.address=new}
                    }
                }
                XCTAssertFalse(try restored.matches(compile(changed,version:12),compilerContractVersion:12),"\(id) \(mutation)")
            }
        }
    }
    func testHistoricalProjectionOmitsSchedulerAndRefusesItBeforeCompilerSix()throws {
        for (schema,version) in [(3,3),(4,4),(5,4),(6,5)] {
            var old=try WorkflowExamples.preparedParallelClassifier(device:"cpu");old.schemaVersion=schema
            let plan=try compile(old,version:version),archive=try WorkflowArchivePlan(plan,compilerContractVersion:version)
            XCTAssertNil(try object(archive)["schedulerRecipe"])
            let bytes=try WorkflowArchiveCodec.encodeCapture(capture(old,version:version))
            XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("schedulerRecipe"))
            XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(bytes)),bytes)
            let explicit=try compile(WorkflowExamples.preparedStepLRClassifier(device:"cpu"),version:12)
            XCTAssertThrowsError(try WorkflowArchivePlan(explicit,compilerContractVersion:version))
        }
        var implicit=try WorkflowExamples.preparedParallelClassifier(device:"cpu");implicit.schemaVersion=7
        let plan=try compile(implicit,version:6)
        let archive=try WorkflowArchivePlan(plan,compilerContractVersion:6)
        XCTAssertNil(try object(archive)["schedulerRecipe"])
        XCTAssertTrue(try archive.matches(plan,compilerContractVersion:6))
        XCTAssertNoThrow(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(capture(implicit,version:6))))
    }
    func testOlderCaptureCodecRefusesInjectedOptionalRecipe()throws {
        let explicit=try WorkflowArchivePlan(compile(WorkflowExamples.preparedLinearLRClassifier(device:"cpu"),version:12),compilerContractVersion:12)
        let recipe=try XCTUnwrap(object(explicit)["schedulerRecipe"])
        for (schema,version) in [(3,3),(4,4),(5,4),(6,5)] {
            var old=try WorkflowExamples.preparedParallelClassifier(device:"cpu");old.schemaVersion=schema
            var raw=try object(capture(old,version:version))
            var plan=try XCTUnwrap(raw["plan"] as? [String:Any]);plan["schedulerRecipe"]=recipe;raw["plan"]=plan
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw)),"schema\(schema)/compiler\(version)")
        }
    }
    func testNewCaptureRefusesMissingOrUnknownRecipeFieldsAndWrongSourceDomain()throws {
        let document=try WorkflowExamples.preparedLinearLRClassifier(device:"cpu")
        let valid=try object(capture(document,version:12))
        for mutation in ["missingConfiguration","missingConfig","unknownConfiguration","unknownControl","missingControl","missingOptimizer","wrongDomain"] {
            var raw=valid
            var plan=try XCTUnwrap(raw["plan"] as? [String:Any])
            var recipe=try XCTUnwrap(plan["schedulerRecipe"] as? [String:Any])
            if mutation == "missingConfiguration" {recipe.removeValue(forKey:"configuration")}
            if ["missingConfig","unknownConfiguration","unknownControl","missingControl"].contains(mutation) {
                var configuration=try XCTUnwrap(recipe["configuration"] as? [String:Any])
                if mutation == "missingConfig" {configuration.removeValue(forKey:"config")}
                if mutation == "unknownConfiguration" {configuration["unreviewedPolicy"]=true}
                if mutation == "unknownControl" {
                    var controls=try XCTUnwrap(configuration["config"] as? [String:Any]);controls["unreviewedControl"]=0.5;configuration["config"]=controls
                }
                if mutation == "missingControl" {
                    var controls=try XCTUnwrap(configuration["config"] as? [String:Any]);controls.removeValue(forKey:"start_factor");configuration["config"]=controls
                }
                recipe["configuration"]=configuration
            }
            if mutation == "missingOptimizer" {plan.removeValue(forKey:"optimizerRecipe")}
            if mutation == "wrongDomain" {
                var source=try XCTUnwrap(recipe["source"] as? [String:Any]);source["domain"]="model";recipe["source"]=source
            }
            plan["schedulerRecipe"]=recipe;raw["plan"]=plan
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw)),mutation)
        }
    }

    func testCompletedOutcomeAdmitsOneBoundedRateTraceAndRejectsOversizeOrUnknownFiles() throws {
        let pin = String(repeating: "a", count: 64)
        for count in [1, 4000000, 4000001] {
            let files = [WorkflowArchiveFile(path: "preparation/artifacts/manifest.json", bytes: 20, sha256: pin),
                         WorkflowArchiveFile(path: "training/artifacts/manifest.json", bytes: 20, sha256: pin),
                         WorkflowArchiveFile(path: "training/artifacts/rate-trace.json", bytes: count, sha256: pin)]
            let value = WorkflowArchiveOutcome(projectID: UUID().uuidString, directoryID: UUID().uuidString,
                runID: UUID().uuidString, requestIdentity: pin, status: .completed, terminalPhase: .training,
                startedAt: "2026-10-05T12:00:00.000Z", finishedAt: "2026-10-05T12:00:01.000Z", artifacts: files,
                capture: .init(path: "capture.json", bytes: 100, sha256: pin), preparationManifestSHA256: pin,
                modelManifestSHA256: pin, workerRunID: UUID().uuidString)
            if count > 4000000 { XCTAssertThrowsError(try WorkflowArchiveCodec.encodeOutcome(value)); continue }
            let bytes = try WorkflowArchiveCodec.encodeOutcome(value)
            XCTAssertEqual(try WorkflowArchiveCodec.decodeOutcome(bytes), value)
            var raw = try object(value)
            var changedFiles = try XCTUnwrap(raw["artifacts"] as? [[String: Any]])
            changedFiles[2]["path"] = "training/artifacts/unreviewed_trace.json"
            raw["artifacts"] = changedFiles
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeOutcome(JSONSerialization.data(withJSONObject: raw)))
        }
    }

}
