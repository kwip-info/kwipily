import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowArchiveWriterTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func fixture(block:Bool=false)->WorkflowDocument {
        let base=ExampleFlows.parallelClassifier
        let source=CompositeExamples.parallelClassifier
        var nodes:[WorkflowNode]=[
            .init(address:address(.data,"input"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:address(.data,"cast"),payload:.dataOperator(operatorID:"kwip.data.cast",config:["columns":.object(["a":.string("float64"),"b":.string("float64"),"label":.string("int64")])])),
            .init(address:address(.data,"split"),payload:.dataOperator(operatorID:"kwip.data.split",config:["seed":.integer(42),"validationPercent":.integer(20),"stratifyColumn":.string("label")])),
            .init(address:address(.data,"fit"),payload:.dataOperator(operatorID:"kwip.data.standardScalerFit",config:["columns":.array([.string("a"),.string("b")])])),
            .init(address:address(.data,"training"),payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:address(.data,"validation"),payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:address(.control,"model"),payload:.modelDefinition),
            .init(address:address(.control,"mapping"),payload:.trainingData(config:["featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("label")])),
            .init(address:address(.control,"setup"),payload:.runSetup(config:["device":.string("cpu"),"epochs":.integer(3),"batchSize":.integer(32),"seed":.integer(17),"learningRate":.number(0.02)])),
            .init(address:address(.control,"train"),payload:.train(config:[:]))]
        let primitives=block ? source.root.nodes : base.nodes
        nodes += primitives.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelPrimitive(type:$0.type,configurationVersion:block ? (source.root.configurationVersion ?? 1) : base.schemaVersion,config:$0.config))}
        if block {nodes += source.root.instances.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelBlock(reference:$0.definition,overrides:$0.overrides))}}
        var edges=[edge(.data,"input","table",.data,"cast","table",.table),edge(.data,"cast","table",.data,"split","table",.table),edge(.data,"split","training",.data,"fit","training",.dataset),edge(.data,"split","training",.data,"training","table",.dataset),edge(.data,"split","validation",.data,"validation","table",.dataset),edge(.data,"fit","state",.data,"training","state",.fittedState),edge(.data,"fit","state",.data,"validation","state",.fittedState),edge(.data,"training","table",.control,"mapping","training",.table),edge(.data,"validation","table",.control,"mapping","validation",.table),edge(.model,"output","definition",.control,"model","source",.modelDefinition),edge(.control,"model","model",.control,"train","model",.modelDefinition),edge(.control,"mapping","data",.control,"train","data",.trainingData),edge(.control,"setup","configuration",.control,"train","configuration",.runConfiguration)]
        edges += (block ? source.root.edges : base.edges).map {.init(id:$0.id,from:.init(address:address(.model,$0.source),port:$0.sourcePort),to:.init(address:address(.model,$0.target),port:$0.targetPort),kind:.tensor)}
        return WorkflowDocument(projectID:UUID().uuidString,revision:7,name:"Measured workflow",nodes:nodes,edges:edges,definitions:block ? source.definitions : [])
    }
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    /// Captured historical descriptors support integrity replay, never new authoring.
    private func dataRegistry() throws -> DataFlowOperatorRegistry {
        var captured=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let retired=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/retired-data-operators-v1.json"))
        XCTAssertEqual(captured.packageID,retired.packageID)
        XCTAssertTrue(Set(captured.operators.map(\.id)).isDisjoint(with:retired.operators.map(\.id)))
        captured.operators += retired.operators
        return captured
    }
    @MainActor private func setup() throws -> (URL, WorkflowAuthoringStore, RuntimeLocation) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let library = ProjectLibrary(directory: root.appendingPathComponent("projects"))
        let source = fixture(block: true)
        _ = try library.saveSource(.workflow(source), id: source.projectID)
        let runtime = try makeRuntime(root.appendingPathComponent("runtime"))
        return (root, try WorkflowAuthoringStore(library: library, projectID: source.projectID,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry()), runtime)
    }
    private func makeRuntime(_ root: URL) throws -> RuntimeLocation {
        for name in [".venv/bin", "runner", "scripts", "Sources/KwipMLCore/Resources"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        try Data("fixture interpreter; never executed".utf8).write(to: root.appendingPathComponent(".venv/bin/python"))
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.appendingPathComponent(".venv/bin/python").path)
        for name in ["runner/main.py", "runner/engine.py", "scripts/runtime_launch.py"] {
            try Data("# source fixture\n".utf8).write(to: root.appendingPathComponent(name))
        }
        try JSONEncoder().encode(ComponentRegistry.builtIn()).write(to: root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))
        try JSONEncoder().encode(dataRegistry()).write(to: root.appendingPathComponent("runner/dataflow-operators.json"))
        return try RuntimeLocation.resolve(resources: nil, developmentRoot: root, allowDevelopment: true)
    }
    @MainActor private struct Run {
        let root:URL
        let owner:WorkflowAuthoringStore
        let session:WorkflowExecutionSession
        let directory:URL
        let source:URL
        let exit:(Int32,String)->Void
    }
    @MainActor private func begun(secondTrain:Bool=false) async throws -> Run {
        let(root,owner,runtime) = try setup()
        let terminal = address(.control,secondTrain ? "second-train" : "train")
        if secondTrain {
            XCTAssertNotNil(owner.apply(.insert(.init(address:terminal,payload:.train(config:[:]))),lease:owner.lease()))
            for edge in owner.document.edges.filter({$0.to.address == address(.control,"train")}) {
                XCTAssertNotNil(owner.apply(.connect(from:edge.from,to:.init(address:terminal,port:edge.to.port)),lease:owner.lease()))
            }
        }
        let request=try capturedHistoricalRequest(owner:owner,train:terminal,runtime:runtime)
        let source = root.appendingPathComponent("private-source.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n3,4,0\n4,5,1\n".utf8).write(to:source)
        let directory = root.appendingPathComponent(UUID().uuidString)
        var exit:((Int32,String)->Void)?
        let session = WorkflowExecutionSession(request:request,owner:owner,
            sources:[address(.data,"input"):try WorkflowCapturedSource.capture(source)],directory:directory,
            reservation:.init(acquire:{UUID()},release:{_ in}),launch:{_,_,_,_,finish in
                exit=finish;return .init(cancel:{})
            },onTerminal:{_ in})
        XCTAssertTrue(session.start(),session.statusMessage)
        let deadline=Date().addingTimeInterval(5)
        while exit == nil && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
        return Run(root:root,owner:owner,session:session,directory:directory,source:source,exit:try XCTUnwrap(exit))
    }
    @MainActor private func capturedHistoricalRequest(owner:WorkflowAuthoringStore,train:WorkflowNodeAddress,runtime:RuntimeLocation)throws->WorkflowExecutionRequest {
        // Simulate an already-captured historical request. Current capture must reject
        // the original retired source; archive tests must never upgrade its bytes.
        XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:train,runtime:runtime))
        let plan=try WorkflowCompiler.compileHistoricalEvidence(owner.document,train:train,
            modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:2)
        return WorkflowExecutionRequest(lease:owner.lease(),document:owner.document,authoringDocument:owner.document,
            experiment:nil,experimentInputs:nil,train:train,plan:plan,
            modelRegistrySHA256:try WorkflowArchiveCodec.registryHash(owner.modelRegistry),
            dataRegistrySHA256:try WorkflowArchiveCodec.registryHash(owner.dataRegistry),runtime:runtime,
            runtimeEvidence:try WorkflowRuntimeEvidence.capture(runtime),requestIdentity:String(repeating:"a",count:64),
            priorCache:nil,gradientArguments:[],checkpointParent:nil,executionBudget:nil,sampleCapture:nil,preparationCacheLimitBytes:nil)
    }
    func testImmutableSourceWriteLimitIsExplicitAndDirectoryReservationIsExclusive() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source = Data(repeating:65,count:10_000_000)
        XCTAssertThrowsError(try DiagnosticFileIO.writeImmutable(source,root:root,parts:["default.csv"]))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("default.csv").path))
        XCTAssertNoThrow(try DiagnosticFileIO.writeImmutable(source,root:root,parts:["source.csv"],limit:10_000_000))
        XCTAssertEqual(try DiagnosticFileIO.read(root:root,parts:["source.csv"],limit:10_000_000),source)
        XCTAssertThrowsError(try DiagnosticFileIO.writeImmutable(Data(repeating:65,count:10_000_001),root:root,parts:["too-large.csv"],limit:10_000_000))
        let name = UUID().uuidString
        try DiagnosticFileIO.createExclusiveDirectory(root:root,name:name)
        XCTAssertThrowsError(try DiagnosticFileIO.createExclusiveDirectory(root:root,name:name))
        let alias = UUID().uuidString
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent(alias),withDestinationURL:root.appendingPathComponent(name))
        XCTAssertThrowsError(try DiagnosticFileIO.createExclusiveDirectory(root:root,name:alias))
        XCTAssertThrowsError(try DiagnosticFileIO.writeImmutable(Data("x".utf8),root:root,parts:[alias,"capture.json"]))
    }

    @MainActor func testCapturePinsPortablePlanRegistriesSourcesAndSeparateIdentities() async throws {
        let run = try await begun();defer{run.exit(1,"fixture cleanup");try? FileManager.default.removeItem(at:run.root)}
        let raw = try Data(contentsOf:run.directory.appendingPathComponent("capture.json"))
        let capture = try WorkflowArchiveCodec.decodeCapture(raw)
        XCTAssertEqual(capture.projectID,run.owner.projectID)
        XCTAssertEqual(capture.directoryID,run.directory.lastPathComponent)
        XCTAssertEqual(capture.runID,run.session.runID.uuidString)
        XCTAssertNotEqual(capture.directoryID,capture.runID)
        XCTAssertEqual(capture.requestIdentity,run.session.request.requestIdentity)
        XCTAssertEqual(capture.train,address(.control,"train"))
        XCTAssertTrue(try capture.plan.matches(run.session.request.plan, compilerContractVersion: 1))
        XCTAssertEqual(capture.modelRegistrySHA256,run.session.request.modelRegistrySHA256)
        XCTAssertEqual(capture.dataRegistrySHA256,run.session.request.dataRegistrySHA256)
        XCTAssertEqual(capture.sources.map(\.address),run.session.request.plan.sourceAddresses)
        for file in [capture.workflow,capture.modelRegistry,capture.dataRegistry,capture.runtimeEvidence] {
            let bytes = try Data(contentsOf:run.directory.appendingPathComponent(file.path))
            XCTAssertEqual(bytes.count,file.bytes);XCTAssertEqual(WorkflowPhaseArtifacts.hash(bytes),file.sha256)
        }
        for file in capture.sources {
            let bytes = try Data(contentsOf:run.directory.appendingPathComponent(file.path))
            XCTAssertEqual(bytes.count,file.bytes);XCTAssertEqual(WorkflowPhaseArtifacts.hash(bytes),file.sha256)
            XCTAssertFalse(file.path.hasPrefix("/"));XCTAssertTrue(file.path.hasPrefix("inputs/source-"))
        }
        XCTAssertFalse(String(decoding:raw,as:UTF8.self).contains(run.source.path))
        XCTAssertNil(capture.plan.settings.preparedTraining)
    }
    @MainActor func testExplicitSecondTrainIsRetainedInCaptureAndPortablePlan() async throws {
        let run = try await begun(secondTrain:true);defer{run.exit(1,"cleanup");try? FileManager.default.removeItem(at:run.root)}
        let capture = try WorkflowArchiveCodec.decodeCapture(Data(contentsOf:run.directory.appendingPathComponent("capture.json")))
        XCTAssertEqual(capture.train,address(.control,"second-train"))
        XCTAssertEqual(capture.plan.train,capture.train)
        XCTAssertEqual(capture.revision,run.session.request.document.revision)
        XCTAssertTrue(try capture.plan.matches(run.session.request.plan, compilerContractVersion: 1))
    }

    @MainActor func testFailedTerminalPublishesV2OnceWithPinnedCaptureAndOriginalError() async throws {
        let run = try await begun();defer{try? FileManager.default.removeItem(at:run.root)}
        run.exit(1,"raw worker failure")
        let url = run.directory.appendingPathComponent("outcome.json")
        let bytes = try Data(contentsOf:url),record = try WorkflowArchiveCodec.decodeOutcome(bytes)
        XCTAssertEqual(record.schemaVersion,2);XCTAssertEqual(record.status,.failed)
        XCTAssertEqual(record.terminalPhase,.preparing);XCTAssertEqual(record.error,"raw worker failure")
        XCTAssertEqual(record.projectID,run.owner.projectID);XCTAssertEqual(record.runID,run.session.runID.uuidString)
        let capture = try XCTUnwrap(record.capture)
        XCTAssertEqual(capture.sha256,WorkflowPhaseArtifacts.hash(try Data(contentsOf:run.directory.appendingPathComponent("capture.json"))))
        XCTAssertTrue(record.artifacts.isEmpty);XCTAssertNil(record.modelManifestSHA256);XCTAssertNil(record.workerRunID)
        XCTAssertLessThanOrEqual(record.startedAt,record.finishedAt)
        run.exit(0,"")
        XCTAssertEqual(try Data(contentsOf:url),bytes,"Duplicate exit cannot replace immutable terminal evidence")
        XCTAssertNotNil(run.owner.apply(.rename(address:address(.data,"input"),name:"Current edit"),lease:run.owner.lease()))
        let archived = try WorkflowCodec.decode(Data(contentsOf:run.directory.appendingPathComponent("workflow.json")))
        XCTAssertEqual(archived,run.session.request.document)
        XCTAssertNotEqual(archived,run.owner.document)
    }
    @MainActor func testCancelledPreparationRecordsActualTerminalPhaseWithoutModelClaims() async throws {
        let run = try await begun();defer{try? FileManager.default.removeItem(at:run.root)}
        run.session.cancel();run.exit(0,"")
        let record = try WorkflowArchiveCodec.decodeOutcome(Data(contentsOf:run.directory.appendingPathComponent("outcome.json")))
        XCTAssertEqual(record.status,.cancelled);XCTAssertEqual(record.terminalPhase,.preparing)
        XCTAssertNotNil(record.capture);XCTAssertNil(record.modelManifestSHA256);XCTAssertNil(record.workerRunID)
    }
    @MainActor func testExistingTerminalReceiptIsPreservedAndPublicationFailureIsVisible() async throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let request=try capturedHistoricalRequest(owner:owner,train:address(.control,"train"),runtime:runtime)
        let source = root.appendingPathComponent("source.csv");try Data("a,b,label\n1,2,0\n".utf8).write(to:source)
        let directory = root.appendingPathComponent(UUID().uuidString)
        var finish:((Int32,String)->Void)?,outcome:WorkflowExecutionOutcome?
        let session = WorkflowExecutionSession(request:request,owner:owner,sources:[address(.data,"input"):try WorkflowCapturedSource.capture(source)],directory:directory,
            reservation:.init(acquire:{UUID()},release:{_ in}),launch:{_,_,_,_,exit in finish=exit;return .init(cancel:{})},onTerminal:{outcome=$0})
        XCTAssertTrue(session.start())
        let deadline=Date().addingTimeInterval(5)
        while finish == nil && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
        XCTAssertNotNil(finish)
        let marker = Data("existing evidence must survive".utf8)
        try marker.write(to:directory.appendingPathComponent("outcome.json"))
        finish?(1,"worker failed")
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("outcome.json")),marker)
        XCTAssertNotNil(outcome?.recordError);XCTAssertEqual(outcome?.status,.failed)
    }
}
