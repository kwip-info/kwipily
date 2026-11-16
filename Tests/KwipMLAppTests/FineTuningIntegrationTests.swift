import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FineTuningIntegrationTests:XCTestCase {
    private func wait(_ predicate:()->Bool)async throws {
        let deadline=Date().addingTimeInterval(30)
        while Date()<deadline {if predicate() {return};try await Task.sleep(for:.milliseconds(20))}
        throw WorkflowArchiveError("Timed out waiting for actual fine-tuning.")
    }
    func testActualPreparedTextTrainGenerationSaveAndHistoryWithCaptureOff()async throws {try await exercise(mode:"full",capture:false)}
    func testSelectedLayersCaptureMetricsAndRequiredPredictionRows()async throws {try await exercise(mode:"selected",capture:true)}
    func testLoRAAdapterMergeExportAndGeneration()async throws {try await exercise(mode:"lora",capture:true)}
    func testCompleteOwnedTablesAboveEightMBSurviveCaptureOffAndHistory()async throws {try await exercise(mode:"full",capture:false,largeTable:true)}
    func testCancellationPublishesFinalStatusAndPreservesBaseBeforeReturningIdle()async throws {try await exercise(mode:"full",capture:false,cancelAfterStart:true)}
    func testProjectPredictionOmissionWithHistoryEnabledPreservesModelGenerationAndSave()async throws {try await exercise(mode:"full",capture:true,omitOptionalPredictions:true)}
    func testConnectedPredictionViewStillReceivesRequiredRowsWhenOptionalCaptureIsOff()async throws {try await exercise(mode:"selected",capture:true,omitOptionalPredictions:true,requiresPredictionRows:true)}
    private func exercise(mode:String,capture:Bool,largeTable:Bool=false,cancelAfterStart:Bool=false,omitOptionalPredictions:Bool=false,requiresPredictionRows:Bool=false)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_FOLDER"] else {throw XCTSkip("Needs explicit local tiny model and actual isolated runtime")}
        let reviewPath=largeTable ? ProcessInfo.processInfo.environment["KWIPML_UAT_PIPELINE_ROOT"]:nil
        let root:URL
        if let reviewPath {
            let requested=URL(fileURLWithPath:reviewPath)
            guard requested.lastPathComponent=="pipeline-native-review",!FileManager.default.fileExists(atPath:requested.path) else {throw WorkflowArchiveError("Use a new, explicitly named pipeline-native-review fixture directory.")}
            try FileManager.default.createDirectory(at:requested,withIntermediateDirectories:true)
            root=requested.resolvingSymlinksInPath()
        } else {root=try canonicalBackendTemporaryDirectory()}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer {host.cancel();if reviewPath==nil && !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        if largeTable {
            let csv=root.appendingPathComponent("large-documents.csv")
            let padding=String(repeating:"x",count:13_000)
            // Distinct unused values keep this a real >8 MB transfer fixture even
            // when immutable capture deduplicates repeated chunks.
            let content="text,padding\n"+(0..<700).map {"hello world .,\($0):\(padding)\n"}.joined()
            try Data(content.utf8).write(to:csv)
            XCTAssertTrue(host.bindings.setPath(csv.path,for:FineTuningPracticeExample.source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
            XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.partition]),lease:host.owner.lease()))
            let partition=try WorkflowNodeEditingAuthority(owner:host.owner,address:FineTuningPracticeExample.partition)
            var config=partition.configuration;config["validationPercent"]=9;config["testPercent"]=1
            XCTAssertTrue(partition.configure(config,expected:partition.configuration))
        }
        try host.owner.setRunHistoryPolicy(capture ? .all:.none)
        if omitOptionalPredictions {XCTAssertTrue(workspace.setProjectSampleCapture(.init(activationValues:false,predictionRows:false),projectID:host.owner.projectID))}
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.train]),lease:host.owner.lease()))
        let trainContext=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.train)
        let expected=trainContext.configuration;var configuration=expected
        var fine=(try FineTuningConfiguration.decode(expected))!.values
        fine["mode"] = .string(mode)
        if mode=="selected" {fine["groups"] = .array([.string("block:0")])}
        if mode=="lora" {fine["targets"] = .array([.string("attentionInput")]);fine["rank"]=2;fine["alpha"]=4}
        configuration["fineTuning"] = .object(fine)
        if cancelAfterStart,case .object(var run)=configuration["runSetup"] {run["epochs"]=1000;configuration["runSetup"] = .object(run)}
        if largeTable,case .object(var run)=configuration["runSetup"] {run["batchSize"]=64;configuration["runSetup"] = .object(run)}
        XCTAssertTrue(trainContext.authority.configure(configuration,expected:expected),trainContext.statusMessage)
        let runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
        try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:FineTuningPracticeExample.model,runtime:runtime,policy:policy)
        let loaded=try XCTUnwrap(host.modelLoader.reference(FineTuningPracticeExample.model))
        guard case .object(let inventory)=loaded.metadata["training"],case .array(let supported)=inventory["supportedModes"] else {return XCTFail("Actual task inventory must accompany the verified model")}
        XCTAssertTrue(supported.contains(.string(mode)))
        let view=WorkflowNodeAddress(domain:.data,localID:"fine-tuning-view"),generatedView=WorkflowNodeAddress(domain:.data,localID:"generated-view")
        if capture && (!omitOptionalPredictions || requiresPredictionRows) {
            XCTAssertTrue(host.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:host.owner.lease()))
            XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:FineTuningPracticeExample.train,port:"predictions"),to:.init(address:view,port:"table")),lease:host.owner.lease()))
            XCTAssertTrue(host.canvas.apply(.insert(.init(address:generatedView,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:host.owner.lease()))
            XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:FineTuningPracticeExample.generate,port:"table"),to:.init(address:generatedView,port:"table")),lease:host.owner.lease()))
        }
        let destination=root.appendingPathComponent("saved-model")
        XCTAssertFalse(host.run(),"A connected Save model needs its destination before whole-flow training")
        XCTAssertTrue(host.statusMessage.contains("Choose Save model’s destination"),host.statusMessage)
        XCTAssertFalse(host.hasActiveOperation)
        XCTAssertTrue(host.modelLoader.trainingRecords.isEmpty,"Missing write access must refuse before training starts")
        XCTAssertTrue(host.modelDestinations.set(destination:destination,for:FineTuningPracticeExample.save,lease:host.owner.lease()))
        XCTAssertTrue(host.owner.select(.init(),lease:host.owner.lease()))
        // A newly selected Results table can claim a bounded read just as the
        // trained model becomes available. Automatic consumers must wait for it.
        var completionQuery:TableViewQuerySession?
        var completionReadStarted=false
        if mode=="lora" {
            let publish=host.modelLoader.onBindingChange
            host.modelLoader.onBindingChange={ [weak host] in
                publish?()
                guard let host,!completionReadStarted,!host.modelLoader.isBusy,host.modelLoader.trainingRecords.count==1 else{return}
                completionReadStarted=true
                completionQuery=host.automationTableViewSession(for:.init(address:FineTuningPracticeExample.partition,port:"training"))
                XCTAssertTrue(completionQuery?.load(offset:0,search:"",sort:.init()) == true,"Actual captured table read must contend with the consumer handoff")
            }
        }
        XCTAssertTrue(host.run(),host.statusMessage)
        if cancelAfterStart {
            try await wait {host.modelLoader.trainingPhase=="training"}
            host.cancel()
            try await wait {!workspace.isBusy && !host.owner.isBusy && !host.modelLoader.isBusy}
            XCTAssertEqual(host.modelLoader.trainingPhase,"cancelled")
            XCTAssertFalse(host.hasActiveOperation,"Cancellation must clear pending model consumers")
            XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
            XCTAssertEqual(host.modelLoader.status,"Model operation cancelled.")
            XCTAssertEqual(host.statusMessage,host.modelLoader.status,"Terminal callback must publish the final status, not the last batch")
            XCTAssertTrue(host.modelLoader.trainingRecords.isEmpty)
            XCTAssertTrue(host.modelLoader.trainingMetrics.isEmpty)
            XCTAssertTrue(host.modelLoader.trainingProgress.points.isEmpty)
            XCTAssertNil(host.modelLoader.failure(FineTuningPracticeExample.train))
            for file in loaded.files {try WorkflowPreviewToolContract.verifyFile(file:file,directory:loaded.directory(root:root),limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}
            return
        }
        try await wait {host.modelLoader.trainingRecords.count==1 || host.modelLoader.failure(FineTuningPracticeExample.train) != nil || (!workspace.isBusy && !host.modelLoader.isBusy && host.currentDataPreparation?.phase == .failed)}
        try await wait {!workspace.isBusy && !host.owner.isBusy && !host.modelLoader.isBusy && !host.hasActiveOperation}
        if mode=="lora" {
            XCTAssertTrue(completionReadStarted)
            XCTAssertNotNil(completionQuery?.page,"The bounded reader must finish before automatic generation")
        }
        _=try XCTUnwrap(host.modelLoader.generationRecord,"Whole-flow imported training must run its unselected Generate consumer")
        XCTAssertNotNil(host.externalModelSaveReceipts[FineTuningPracticeExample.save],"The same flow must publish Save model and return idle")
        XCTAssertTrue(host.owner.selection.nodeAddresses.isEmpty,"Completion must not select a consumer to make it run")
        if capture && (!omitOptionalPredictions || requiresPredictionRows) {
            do {try await wait {host.predictionPreparation?.phase == .failed || (host.predictionPreparation?.phase == .completed && !workspace.isBusy)}}
            catch {throw WorkflowArchiveError("Prediction boundary: "+host.statusMessage+" / "+(host.predictionPreparation?.statusMessage ?? "No preparation launched"))}
            XCTAssertEqual(host.predictionPreparation?.phase,.completed,host.statusMessage+" / "+(host.predictionPreparation?.statusMessage ?? "No preparation"))
            guard host.predictionPreparation?.phase == .completed else {return}
            XCTAssertNotNil(host.snapshot(for:view),"A connected View must receive imported Train's complete predictions")
            XCTAssertNotNil(host.snapshot(for:generatedView),"The unselected Generate consumer must publish its connected View")
        }
        let record=try XCTUnwrap(host.modelLoader.trainingRecords.first,host.statusMessage+" / "+host.modelLoader.status)
        let result=try await Task.detached {try record.report(root:root)}.value
        XCTAssertEqual(result["completedEpoch"],1)
        XCTAssertEqual(record.progressFile != nil,capture)
        XCTAssertEqual(host.modelLoader.predictionArtifact(FineTuningPracticeExample.train) != nil,capture && (!omitOptionalPredictions || requiresPredictionRows))
        if !capture {XCTAssertTrue(host.modelLoader.trainingProgress.points.isEmpty)}
        XCTAssertEqual(host.modelLoader.trainingMetrics.count,1)
        XCTAssertTrue(FileManager.default.fileExists(atPath:record.directory(root).appendingPathComponent("input/training.table.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:record.directory(root).appendingPathComponent("source").path))
        if largeTable {
            let captured=record.directory(root).appendingPathComponent("input/training.table.json")
            let request=try WorkflowToolRequest.decode(DiagnosticFileIO.read(root:record.directory(root),parts:["request.json"],limit:256_000))
            let segments=request.inputFiles.filter {ModelAssetToolContract.isSegment($0.path)}
            let retainedBytes=segments.isEmpty ? try FileManager.default.attributesOfItem(atPath:captured.path)[.size] as? Int : segments.reduce(0){$0+$1.bytes}
            XCTAssertGreaterThan(try XCTUnwrap(retainedBytes),8_000_000)
            try ModelAssetToolContract.verifyCaptureVersion(request,directory:record.directory(root))
            let pin=try XCTUnwrap(request.inputFiles.first(where:{$0.path=="input/training.table.json"})?.sha256)
            var held=true
            let dependencies=segments.map {WorkflowArchiveFile(path:String($0.path.dropFirst(6)),bytes:$0.bytes,sha256:$0.sha256)}
            let table=TableViewQuerySession(artifact:.init(indexURL:captured,indexSHA256:pin,sourceSHA256:pin,inputKind:"table",retainedSegments:dependencies,workloadPolicy:request.executionBudget.resources?.workloadPolicy),workspace:root,accountingRoot:root,projectID:host.owner.projectID,runtime:runtime,reservation:.init(acquire:{held ? nil:UUID()},release:{_ in}),accepts:{true})
            XCTAssertFalse(table.load(offset:100,search:"hello",sort:.init()))
            XCTAssertTrue(table.waitingForExecution)
            held=false;XCTAssertTrue(table.resumeAfterExecution())
            try await wait {!table.isBusy}
            XCTAssertEqual(table.page?.totalRows,630,table.statusMessage)
            XCTAssertEqual(table.page?.offset,100)
            XCTAssertEqual(table.page?.rows.count,100)
            XCTAssertFalse(table.waitingForExecution)
            var inspected:CapturedCellPage?
            table.readFullCell(row:0,column:1){inspected=$0}
            try await wait {inspected != nil}
            let cell=try XCTUnwrap(inspected?.text)
            XCTAssertTrue(cell.hasSuffix(String(repeating:"x",count:13_000)))
            XCTAssertTrue((0..<700).contains(Int(cell.split(separator:":",maxSplits:1)[0]) ?? -1))
        }
        let trained=try XCTUnwrap(host.modelLoader.reference(FineTuningPracticeExample.train),host.modelLoader.status)
        XCTAssertNotNil(host.snapshot(for:FineTuningPracticeExample.train)?.ports["model"])
        XCTAssertNotEqual(trained.identity,host.modelLoader.reference(FineTuningPracticeExample.model)?.identity)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.save]),lease:host.owner.lease()))
        let saveContext=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.save)
        XCTAssertEqual(saveContext.saveModelConfiguration?.content,.full,"Inherited contents must export the displayed standalone model without rewriting source")
        XCTAssertTrue(FileManager.default.fileExists(atPath:destination.appendingPathComponent("model.safetensors").path),host.modelLoader.status)
        if mode=="lora" {
            let freshContext=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.save)
            var config=freshContext.configuration;config["content"] = .string("adapter")
            XCTAssertTrue(freshContext.authority.configure(config,expected:freshContext.configuration))
            XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("adapter-only"),for:FineTuningPracticeExample.save,lease:host.owner.lease()))
            XCTAssertTrue(host.saveModel(address:FineTuningPracticeExample.save),host.statusMessage)
            try await wait {!host.modelLoader.isBusy}
            XCTAssertTrue(FileManager.default.fileExists(atPath:root.appendingPathComponent("adapter-only/adapter_model.safetensors").path),host.modelLoader.status)
            XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("adapter-only/model.safetensors").path))
            XCTAssertFalse(host.saveModel(address:FineTuningPracticeExample.save),"An existing destination must refuse before export begins")
            XCTAssertTrue(host.statusMessage.contains("Choose a new destination"),host.statusMessage)
            XCTAssertFalse(host.modelLoader.isBusy)
            XCTAssertFalse(host.hasActiveOperation,"A refused imported export must release busy state")
            XCTAssertEqual(host.modelLoader.trainingRecords.count,1,"Saving must never retrain the model")
        }
        let reloaded=try ModelLoaderStore(owner:host.owner,root:root,accountingRoot:root)
        try await wait {!reloaded.trainingRecords.isEmpty}
        XCTAssertEqual(reloaded.reference(FineTuningPracticeExample.train)?.identity,trained.identity,reloaded.status)
        XCTAssertTrue(HistoryTableCatalog.rows(fineTuning:reloaded.trainingRecords).contains{$0.target == .fineTuning(record.id)})
        if reviewPath != nil {
            workspace.renameProject("Pipeline review · complete tables")
            _=workspace.saveManagedProject()
            let receipt:[String:String]=["projectID":host.owner.projectID,"root":root.path,"operationID":record.operationID]
            try JSONEncoder().encode(receipt).write(to:root.appendingPathComponent("native-review.json"))
            return
        }
        let corrupted=record.directory(root).appendingPathComponent("output/training/training.json")
        try Data("{}".utf8).write(to:corrupted)
        XCTAssertThrowsError(try record.verify(root:root))
    }
}
