import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientCaptureFormatTests:XCTestCase {
    private let pin=String(repeating:"a",count:64)
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func document(_ task:GradientLearningTask = .combineSmallGroups,selected:Bool=true,cache:Bool=false)throws->WorkflowDocument {
        var doc=try GradientWorkflowExamples.document(task:task,device:"cpu")
        if !selected {
            let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
            guard case .train(var config)=doc.nodes[index].payload else {throw WorkflowPhaseError("Missing Train")}
            config.removeValue(forKey:"gradientPolicy");doc.nodes[index].payload = .train(config:config)
        }
        if cache {
            let i=try XCTUnwrap(doc.nodes.firstIndex {if case .runSetup = $0.payload {return true};return false})
            guard case .runSetup(var config)=doc.nodes[i].payload else {throw WorkflowPhaseError("Missing run setup")}
            config["preparationCachePolicy"] = .string("materialize");doc.nodes[i].payload = .runSetup(config:config)
        }
        return doc
    }
    /// Explicit archive-plan fixture matching GradientArchiveTests' historical format5 shape.
    /// It never downgrades a current gradient lesson or authorizes a worker request.
    private func historicalPlan(_ task:GradientLearningTask,selected:Bool,cache:Bool)throws->WorkflowExecutionPlan {
        var source=try WorkflowExamples.preparedParallelClassifier(device:"cpu");source.schemaVersion=8
        if cache {
            let index=try XCTUnwrap(source.nodes.firstIndex {if case .runSetup=$0.payload {return true};return false})
            guard case .runSetup(var config)=source.nodes[index].payload else {throw WorkflowArchiveError("Missing Run setup")}
            config["preparationCachePolicy"] = .string("materialize");source.nodes[index].payload = .runSetup(config:config)
        }
        var plan=try WorkflowCompiler.compile(source,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:7)
        if selected {
            var requests:[JSONValue]=[]
            if task == .selectedInfluence {
                let input=try XCTUnwrap(plan.model.graph.nodes.first {$0.type=="input"})
                let output=try XCTUnwrap(plan.model.graph.nodes.first {$0.type=="output"})
                let owner=try XCTUnwrap(plan.model.graph.nodes.first {$0.type=="linear"})
                requests=[.object(["id":.string("effect"),"output":.object(["node":.string(output.id),"port":.string("out")]),"partition":.string("validation"),"row_start":0,"row_count":2,"reduction":.string("mean"),"upstream":.null,"targets":.array([.object(["kind":.string("input"),"node":.string(input.id),"port":.string("out")]),.object(["kind":.string("parameter"),"node":.string(owner.id),"parameter":.string("weight")])]),"unavailable":.string("reportUnavailable")])]
            }
            plan.gradientPolicyRecipe = .init(configuration:try .init(values:["accumulation_steps":2,"requests":.array(requests)]),source:.init(domain:.control,localID:"gradient-policy"))
        }
        return plan
    }
    private func capture(_ task:GradientLearningTask = .combineSmallGroups,selected:Bool=true,cache:Bool=false)throws->WorkflowArchiveCapture {
        let p=try historicalPlan(task,selected:selected,cache:cache)
        func file(_ name:String)->WorkflowArchiveFile {.init(path:name,bytes:10,sha256:pin)}
        return .init(schemaVersion:10,compilerContractVersion:9,projectID:p.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,revision:p.revision,train:p.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),sources:p.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},plan:try WorkflowArchivePlan(p,compilerContractVersion:9))
    }
    func testSelectedEmptyRequestsRetainExplicitTrueAndFalseInFormatFive()throws {
        for values in [false,true] {
            var c=try capture();c.captureFormatVersion=5;c.executionBudget=try ExecutionPreferences.factory.resolve().budget;c.derivativeCapture = .init(values:values)
            let data=try WorkflowArchiveCodec.encodeCapture(c),decoded=try WorkflowArchiveCodec.decodeCapture(data)
            XCTAssertEqual(decoded.derivativeCapture,.init(values:values));XCTAssertEqual(decoded.captureFormatVersion,5)
            XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(decoded),data)
        }
    }
    func testSelectedPolicyCannotBypassPermissionViaOlderFormatOrMissingBudget()throws {
        var c=try capture();c.executionBudget=try ExecutionPreferences.factory.resolve().budget
        let formats:[Int?]=[nil,2,3,4]
        for format in formats {
            c.captureFormatVersion=format;c.sampleCapture=format==3 ? .factory:nil
            if format==nil {c.executionBudget=nil}else{c.executionBudget=try ExecutionPreferences.factory.resolve().budget}
            XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(c),"format \(String(describing:format))")
        }
        c.captureFormatVersion=5;c.sampleCapture=nil;c.derivativeCapture = .factory;c.executionBudget=nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(c))
        c.executionBudget=try ExecutionPreferences.factory.resolve().budget;c.derivativeCapture=nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(c))
        var requested=try capture(.selectedInfluence);requested.captureFormatVersion=5;requested.executionBudget=try ExecutionPreferences.factory.resolve().budget;requested.derivativeCapture = .factory
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(requested))
    }
    func testUnselectedPermissionAndMalformedPermissionRefuse()throws {
        var old=try capture(selected:false);old.captureFormatVersion=2;old.executionBudget=try ExecutionPreferences.factory.resolve().budget
        let bytes=try WorkflowArchiveCodec.encodeCapture(old)
        XCTAssertFalse(String(decoding:bytes,as:UTF8.self).contains("derivativeCapture"))
        old.derivativeCapture = .init(values:true)
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(old))
        old.captureFormatVersion=5;XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(old))
        var valid=try capture();valid.captureFormatVersion=5;valid.executionBudget=try ExecutionPreferences.factory.resolve().budget;valid.derivativeCapture = .factory
        let data=try WorkflowArchiveCodec.encodeCapture(valid)
        let policies:[[String:Any]]=[["schemaVersion":1,"values":0],["schemaVersion":1,"values":false,"foreign":true],["schemaVersion":2,"values":true]]
        for policy in policies {
            var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any]);raw["derivativeCapture"]=policy
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw)))
        }
    }
    func testFormatFivePreservesOptionalSampleAndCacheProvenanceIndependently()throws {
        var c=try capture(cache:true);c.captureFormatVersion=5;c.executionBudget=try ExecutionPreferences.factory.resolve().budget;c.derivativeCapture = .factory
        c.sampleCapture = .init(activationValues:false,predictionRows:true);c.preparationCacheLimitBytes=12345
        let decoded=try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(c))
        XCTAssertEqual(decoded.sampleCapture,c.sampleCapture);XCTAssertEqual(decoded.preparationCacheLimitBytes,12345);XCTAssertEqual(decoded.derivativeCapture,.factory)
        c.preparationCacheLimitBytes=0;XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(c))
        var off=try capture();off.captureFormatVersion=5;off.executionBudget=c.executionBudget;off.derivativeCapture = .factory;off.preparationCacheLimitBytes=12345
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(off))
    }
    func testActualWriterRetainsSelectedPermissionWithCurrentFormatEight()async throws {
        let discoveryRoot=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:discoveryRoot)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:discoveryRoot,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let proof=try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence)

        for selected in [true,false] {
            for permission in [false,true] {
                let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
                let doc=try document(selected:selected),library=ProjectLibrary(directory:root.appendingPathComponent("projects"));_ = try library.saveSource(.workflow(doc),id:doc.projectID)
                let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
                let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
                let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,executionBudget:ExecutionPreferences.factory.resolve().budget,derivativeCapture:selected ? .init(values:permission):nil,moduleCompositionEvidence:proof)
                XCTAssertEqual(request.derivativeCapture,selected ? .init(values:permission):nil)
                let directory=root.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                let bytes=Data(try GradientWorkflowExamples.sourceCSV().utf8),source=root.appendingPathComponent("input.csv");try bytes.write(to:source)
                let sources=Dictionary(uniqueKeysWithValues:request.plan.sourceAddresses.map {($0,WorkflowCapturedSource(url:source,sha256:WorkflowArchiveCodec.hash(bytes),byteCount:bytes.count))})
                _ = try WorkflowArchiveCaptureWriter.write(request:request,runID:UUID(),sources:sources,directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
                let data=try Data(contentsOf:directory.appendingPathComponent("capture.json")),saved=try WorkflowArchiveCodec.decodeCapture(data)
                XCTAssertEqual(saved.captureFormatVersion,8);XCTAssertEqual(saved.derivativeCapture,request.derivativeCapture)
                XCTAssertEqual(saved.executionBudget,request.executionBudget)
                if !selected {XCTAssertFalse(String(decoding:data,as:UTF8.self).contains("derivativeCapture"))}
            }
        }
    }
}
