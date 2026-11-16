import Foundation
import KwipMLCore

extension WorkspaceStore {
    @discardableResult func loadPreparationPracticeExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .preparationPractice)
    }
    @discardableResult func loadBackendWorkflowExample(backendID: String) -> Bool {
        loadPreparedWorkflowExample(device: "cpu", kind: .backend(backendID))
    }
    @discardableResult func loadComponentWorkflowExample(_ id: String, device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .component(["data02-calculate","data02-regex-extract","data02-regex-replace"].contains(id) ? FormulaPracticeExample.exampleID:id))
    }
    @discardableResult func loadMaxPool2dWorkflowExample(device:String = "auto") -> Bool {
        loadPreparedWorkflowExample(device:device,kind:.maxPool2d)
    }
    @discardableResult func loadMaxPool1dWorkflowExample(device:String = "auto") -> Bool {
        loadPreparedWorkflowExample(device:device,kind:.maxPool1d)
    }
    @discardableResult func loadSeparableConv2dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .separableConv2d)
    }
    @discardableResult func loadDepthwiseConv2dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .depthwiseConv2d)
    }
    @discardableResult func loadConvTranspose3dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .convTranspose3d)
    }
    @discardableResult func loadConvTranspose2dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .convTranspose2d)
    }
    @discardableResult func loadConvTranspose1dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .convTranspose1d)
    }
    @discardableResult func loadConv3dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .conv3d)
    }
    @discardableResult func loadConv2dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .conv2d)
    }
    @discardableResult func loadConv1dWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .conv1d)
    }
    @discardableResult func loadLazyLinearWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .lazyLinear)
    }
    @discardableResult func loadBilinearWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device: device, kind: .bilinear)
    }
    @discardableResult func loadPreparedWorkflowExample(device: String = "auto") -> Bool {
        loadPreparedWorkflowExample(device:device,kind:.parallel)
    }
    @discardableResult func loadTensorLayoutWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.tensorLayout)
    }
    @discardableResult func loadSoftmaxWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.softmax)
    }
    @discardableResult func loadLayerNormWorkflowExample(normalizeWholeSample:Bool = false,device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.layerNorm(normalizeWholeSample:normalizeWholeSample))
    }
    @discardableResult func loadGELUWorkflowExample(approximateTanh:Bool = false,device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.gelu(approximateTanh:approximateTanh))
    }
    @discardableResult func loadEmbeddingWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.embedding)
    }
    @discardableResult func loadMatrixWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.matrix)
    }
    @discardableResult func loadGroupSplitWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.groupPartition)
    }
    @discardableResult func loadTemporalSplitWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.temporalPartition)
    }
    @discardableResult func loadAggregatedWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.aggregated)
    }
    @discardableResult func loadJoinedWorkflowExample(device:String = "auto")->Bool {
        loadPreparedWorkflowExample(device:device,kind:.joined)
    }
    @discardableResult func loadGuidedWorkflowCourse()->Bool {
        loadPreparedWorkflowExample(device:"auto",kind:.guided)
    }
    func loadPreparedWorkflowExample(device:String,kind:PreparedWorkflowExample,
        resourceURL:(String)->URL? = { name in
            if OCRPracticeExample.resourceNames.contains(name) {let file=name as NSString;return Bundle.module.url(forResource:file.deletingPathExtension,withExtension:file.pathExtension,subdirectory:"OCRPractice") ?? Bundle.module.url(forResource:file.deletingPathExtension,withExtension:file.pathExtension)}
            if name=="media-practice-tone.wav" {return Bundle.module.url(forResource:"media-practice-tone",withExtension:"wav",subdirectory:"MediaPractice") ?? Bundle.module.url(forResource:"media-practice-tone",withExtension:"wav")}
            if FileInputViewPracticeExample.resourceNames.contains(name) {
                let file = name as NSString
                return Bundle.module.url(forResource: file.deletingPathExtension, withExtension: file.pathExtension, subdirectory: "FileInputPractice")
                    ?? Bundle.module.url(forResource: file.deletingPathExtension, withExtension: file.pathExtension)
            }
            if name == "table-output-practice" {return try? TableOutputPracticeExample.resourceURL()}
            if name == "sort-practice" {return try? SortPracticeExample.resourceURL()}
            if name == "unique-practice" {return try? UniquePracticeExample.resourceURL()}
            if name == PreprocessPracticeExample.cleanupObservationsResource {return try? PreprocessPracticeExample.cleanupObservationsURL()}
            if name == ReshapeTablePracticeExample.observationsResource {return try? ReshapeTablePracticeExample.resourceURL()}
            if name == ReshapeTablePracticeExample.wideResource {return try? ReshapeTablePracticeExample.resourceURL(wide:true)}
            if name == ModelOutputPracticeExample.observationsResource {return try? ModelOutputPracticeExample.resourceURL()}
            if UnionPracticeExample.sources.contains(name) {return try? UnionPracticeExample.resourceURL(name)}
            if name == SummarizePracticeExample.observationsResource {return try? SummarizePracticeExample.observationsURL()}
            if name == PreprocessPracticeExample.observationsResource {return try? PreprocessPracticeExample.observationsURL()}
            if name == FormulaTemporalPracticeExample.observationsResource {return try? FormulaTemporalPracticeExample.observationsURL()}
            if name == FormulaPracticeExample.observationsResource {return try? FormulaPracticeExample.observationsURL()}
            if name == PartitionPracticeExample.observationsResource {return try? PartitionPracticeExample.observationsURL()}
            if name.hasPrefix("data02-") { return try? DataEngineeringPractice.resourceURL(name + ".csv") }
            if name == "preparation-practice-observations" { return try? PreparationPracticeExample.observationsURL() }
            return Bundle.module.url(forResource: name, withExtension: "csv")
        })->Bool {
        if let exampleInWindow { return exampleInWindow(device, kind) }
        guard !isBusy else { return false }
        var rollback:[URL]=[]
        var installed=false
        defer {
            if !installed && !rollback.isEmpty {
                for url in rollback.reversed() {try? FileManager.default.removeItem(at:url)}
                refreshProjects()
            }
        }
        do {
            let document = try WorkflowTrainRunSetup.migrate(kind.document(device:device))
            let operators = try workflowDataRegistry()
            if case .backend = kind {
                guard let estimator = document.nodes.first(where: { if case .estimator = $0.payload { return true }; return false }) else { throw DataFlowContractError("Example has no estimator.") }
                _ = try WorkflowEstimatorCompiler.compile(document, estimator: estimator.address, modelRegistry: registry, dataRegistry: operators)
            } else if case .component(let id)=kind,id==PrebuiltModelsPractice.lessonID {
                _=try WorkflowEstimatorCompiler.compile(document,estimator:PrebuiltModelsPractice.estimator,modelRegistry:registry,dataRegistry:operators)
            } else if case .component(let id)=kind,id==DocumentClassificationPracticeExample.exampleID {
                _=try WorkflowEstimatorCompiler.compile(document,estimator:DocumentClassificationPracticeExample.estimator,modelRegistry:registry,dataRegistry:operators)
            } else if case .component(let id)=kind,id==SelfPlayPractice.lessonID {
                _=try SelfPlayPlan.compile(document,address:SelfPlayPractice.train,modelRegistry:registry,dataRegistry:operators)
            } else if case .component(let id)=kind,id==GameEpisodePractice.lessonID {
                _=try GameSessionPlan.compile(document,address:GameEpisodePractice.session,modelRegistry:registry,dataRegistry:operators)
            } else if case .component(let id)=kind,[SpeechTranscriptionPracticeExample.exampleID,AudioClassificationPracticeExample.exampleID].contains(id) {
                _=try WorkflowDataPreparationPlan.compile(document,registry:operators)
            } else if !document.nodes.isEmpty && document.nodes.allSatisfy({ $0.address.domain == .data }) {
                _ = try WorkflowDataPreparationPlan.compile(document, registry: operators)
            } else if let imported=document.nodes.first(where:FineTuningConfiguration.isExternal) {
                _=try WorkflowFineTuningPlan.compile(document,train:imported.address,dataRegistry:operators)
            } else if document.guidedBuild == nil {
                _ = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
                    modelRegistry: registry, dataRegistry: operators,
                    contractVersion:max(2,WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:document.schemaVersion)))
            }
            // Resolve and read every resource before persisting any part of the example.
            // A missing second source must not leave a project with half its bindings.
            let sources=try kind.sourceResources.map { entry -> (WorkflowNodeAddress,String,Data) in
                guard let source=resourceURL(entry.resource) else {
                    throw DataFlowContractError("The bundled example observations are missing. Reinstall the complete application.")
                }
                let bytes=try DiagnosticFileIO.read(root:source.deletingLastPathComponent(),parts:[source.lastPathComponent],limit:10_000_000)
                return (entry.address,entry.resource+".csv",bytes)
            }
            let filePractice: [(String, Data)]
            if case .component(let id)=kind,id==OCRPracticeExample.exampleID {filePractice=try OCRPracticeExample.resourceNames.map {name in guard let url=resourceURL(name) else {throw DataFlowContractError("OCR practice images are missing.")};return (name,try DiagnosticFileIO.read(root:url.deletingLastPathComponent(),parts:[url.lastPathComponent],limit:1_000_000))}}
            else if case .component(let id)=kind,id==DocumentClassificationPracticeExample.exampleID {filePractice=DocumentClassificationPracticeExample.documents}
            else if case .component(let id) = kind, [FileInputViewPracticeExample.exampleID,PrepareMediaPracticeExample.exampleID].contains(id) {
                let resources=id==PrepareMediaPracticeExample.exampleID ? PrepareMediaPracticeExample.resourceNames:FileInputViewPracticeExample.resourceNames
                filePractice = try resources.map { name in
                    guard let url = resourceURL(name) else { throw DataFlowContractError("The bundled File Input practice collection is incomplete. Reinstall the complete application.") }
                    return (name, try DiagnosticFileIO.read(root: url.deletingLastPathComponent(), parts: [url.lastPathComponent], limit: 67_108_864))
                }
            } else { filePractice = [] }
            if hasUnsavedChanges && !saveManagedProject() { return false }
            guard flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition(), learningProjects.beforeProjectTransition() else { return false }
            let projectURL=try projectLibrary.documentURL(id:document.projectID)
            let sourceDirectory=workspaceDirectory.appendingPathComponent("example-sources").appendingPathComponent(document.projectID)
            let bindingDirectory=workspaceDirectory.appendingPathComponent("workflow-sources").appendingPathComponent(document.projectID)
            for url in [projectURL,sourceDirectory,bindingDirectory] {
                guard !FileManager.default.fileExists(atPath:url.path) else {throw DataFlowContractError("Example identity already exists. Try opening a new example.")}
            }
            rollback=[projectURL,sourceDirectory,bindingDirectory]
            _ = try projectLibrary.saveSource(.workflow(document))
            // Persist the app-owned resource separately before installing the new
            // project. Portable documents never contain a machine-specific path.
            let owner = try WorkflowAuthoringStore(library: projectLibrary, projectID: document.projectID,
                modelRegistry: registry, dataRegistry: operators)
            defer { owner.invalidate() }
            let bindings = try WorkflowPrivateSourceBindings(owner: owner, directory: workspaceDirectory)
            // Bundle locations change during upgrades/relocation. Each new example
            // owns its original data bytes in private project storage instead.
            for (address,name,observations) in sources {
                let parts=["example-sources",document.projectID,name]
                try DiagnosticFileIO.writeAtomic(observations,root:workspaceDirectory,parts:parts)
                let durableSource=parts.reduce(workspaceDirectory) {$0.appendingPathComponent($1)}
                guard bindings.setPath(durableSource.path,for:address,lease:owner.lease(),expectedRevision:bindings.revision) else {
                    throw DataFlowContractError(bindings.error ?? "Could not bind the bundled observations.")
                }
            }
            if !filePractice.isEmpty {
                let parts = ["example-sources", document.projectID, "file-input"]
                for (name, bytes) in filePractice { try DiagnosticFileIO.writeAtomic(bytes, root: workspaceDirectory, parts: parts + [name]) }
                let ownedDirectory = parts.reduce(workspaceDirectory) { $0.appendingPathComponent($1) }
                let sources:[WorkflowNodeAddress]
                if case .component(let id)=kind,id==OCRPracticeExample.exampleID {sources=[OCRPracticeExample.source]}
                else if case .component(let id)=kind,id==DocumentClassificationPracticeExample.exampleID {sources=[DocumentClassificationPracticeExample.source]}
                else if case .component(let id)=kind,id==PrepareMediaPracticeExample.exampleID {sources=[PrepareMediaPracticeExample.imageSource,PrepareMediaPracticeExample.audioSource]} else {sources=[FileInputViewPracticeExample.sourceAddress]}
                for source in sources {
                    guard bindings.setPath(ownedDirectory.path,for:source,lease:owner.lease(),expectedRevision:bindings.revision) else {throw DataFlowContractError(bindings.error ?? "Could not bind the bundled media practice collection.")}
                }
            }
            refreshProjects()
            guard beginWorkflowProject(id: document.projectID) else { return false }
            installed=true
            statusMessage = document.nodes.allSatisfy({ $0.address.domain == .data })
                ? "Opened editable data practice · configure the tools, Run Flow, then inspect actual tables in Results. Opening an example does not award completion."
                : "Opened example · prepare data, inspect the architecture, then run the classifier."
            return true
        } catch { statusMessage = "Example could not be opened: \(error.localizedDescription)"; return false }
    }
}
