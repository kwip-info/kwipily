import Foundation
import KwipMLCore

/// Standalone preparation keeps every captured source and cache artifact inside one operation.
enum WorkflowPreparationToolContract {
    static func fileLimit(path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        let parts = try WorkflowArchiveCodec.pathParts(path)
        guard parts.allSatisfy({ $0.utf8.count <= 255 }) else { throw WorkflowFileLeaseError.unsafePath }
        if parts.count==3,parts[0]=="input",parts[1]=="ocr-pack",let asset=OCRPackCatalog.asset(parts[2]) {return asset.bytes}
        if path=="input/preparation-models.json" {return 1_000_000}
        if parts.count==4,parts[0]=="input",parts[1]=="preparation-models",WorkflowArchiveCodec.isHash(parts[2]),["state.json","preparation.json"].contains(parts[3]) {return parts[3]=="state.json" ? 32_000_000:PrebuiltPreparationAttachment.maximumBytes}
        if path=="input/decision-models.json" {return 1_000_000}
        if (4...5).contains(parts.count),parts[0]=="input",parts[1]=="decision-models",WorkflowArchiveCodec.isHash(parts[2]),ModelHubRequest.allowedAssetFile(parts.dropFirst(3).joined(separator:"/")) {return path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000}
        if path=="input/speech-models.json" {return 1_000_000}
        if parts.count==4,parts[0]=="input",parts[1]=="speech-models",WorkflowArchiveCodec.isHash(parts[2]),ModelHubRequest.allowedFile(parts[3]) {return parts[3].hasSuffix(".safetensors") ? 1_400_000_000:8_000_000}
        if path=="input/audio-models.json" {return 1_000_000}
        if parts.count==4,parts[0]=="input",parts[1]=="audio-models",WorkflowArchiveCodec.isHash(parts[2]),ModelHubRequest.allowedFile(parts[3]) {return parts[3].hasSuffix(".safetensors") ? 1_400_000_000:8_000_000}
        switch path {
        case "input/graph.json", "input/reuse/manifest.json": return 2_000_000
        case "input/bindings.json", "input/destinations.json": return 1_000_000
        case "input/node-labels.json": return 65_536
        default:
            if parts.count==3,parts[0]=="input",parts[1]=="captured-tables",parts[2].hasSuffix(".json"),WorkflowArchiveCodec.isHash(String(parts[2].dropLast(5))) {return 50_000_000}
            if parts.count == 4, parts[0] == "input", parts[1] == "sources", WorkflowArchiveCodec.isHash(parts[2]) { return 10_000_000 }
            if parts.count == 4, parts[0] == "input", parts[1] == "file-collections",WorkflowArchiveCodec.isHash(parts[2]),parts[3] == "manifest.json" {return 2_000_000}
            if parts.count == 5, parts[0] == "input", parts[1] == "file-collections",WorkflowArchiveCodec.isHash(parts[2]),parts[3] == "members" {return 67_108_864}
            if parts.count == 3, parts[0] == "input", parts[1] == "reuse" { return workload != nil && PreparationArtifactPolicy.isCapacityFileName(parts[2]) ? Int(workload!.limit(.retainedBytes)) : 50_000_000 }
            throw WorkflowFileLeaseError.unsafePath
        }
    }
    private static func options(_ arguments: [String]) throws -> [String: String] {
        let allowed: Set<String> = ["--dataflow-live-previews","--dataflow-measurements","--defer-database-writes", "--dataflow-sampling", "--dataflow-complete-nodes", "--execute-dataflow", "--destination-bindings", "--source-bindings", "--dataflow-node-labels", "--output", "--dataflow-required-nodes","--dataflow-model-input-nodes","--dataflow-capture-outputs","--dataflow-cache-policy", "--dataflow-cache-limit-bytes", "--dataflow-reuse-manifest", "--dataflow-reuse-manifest-sha256"]
        guard arguments.count <= 128, arguments.count % 2 == 0,
              arguments.allSatisfy({ !$0.isEmpty && !$0.contains("\0") && $0.utf8.count <= 4096 }),
              arguments.reduce(0, { $0 + $1.utf8.count }) <= 131_072 else { throw WorkflowFileLeaseError.unsafePath }
        var result: [String: String] = [:]
        for index in stride(from: 0, to: arguments.count, by: 2) {
            guard allowed.contains(arguments[index]), result[arguments[index]] == nil, !arguments[index + 1].hasPrefix("--") else { throw WorkflowFileLeaseError.unsafePath }
            result[arguments[index]] = arguments[index + 1]
        }
        return result
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?, workload: ExecutionWorkloadPolicy? = nil) throws {
        let args = try options(arguments)
        guard let graph = args["--execute-dataflow"], graph.hasPrefix("/"), graph.hasSuffix("/input/graph.json"),
              !graph.contains("//"), !graph.split(separator: "/").contains("."), !graph.split(separator: "/").contains("..") else { throw WorkflowFileLeaseError.unsafePath }
        let root = directory ?? URL(fileURLWithPath: graph).deletingLastPathComponent().deletingLastPathComponent()
        guard root.isFileURL, UUID(uuidString: root.lastPathComponent)?.uuidString == root.lastPathComponent,
              UUID(uuidString: root.deletingLastPathComponent().lastPathComponent)?.uuidString == root.deletingLastPathComponent().lastPathComponent,
              root.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations",
              graph == root.appendingPathComponent("input/graph.json").path,
              args["--source-bindings"] == root.appendingPathComponent("input/bindings.json").path,
              args["--dataflow-node-labels"] == root.appendingPathComponent("input/node-labels.json").path,
              args["--output"] == root.appendingPathComponent("artifacts").path,
              let policy = args["--dataflow-cache-policy"], ["off", "materialize", "reuse"].contains(policy),
              (3...WorkflowPreviewCapture.Purpose.preparation.captureCount(workload)).contains(files.count), Set(files.map(\.path)).count == files.count else { throw WorkflowFileLeaseError.unsafePath }
        if let raw=args["--dataflow-model-input-nodes"] {
            let nodes=try JSONDecoder().decode([String].self,from:Data(raw.utf8))
            let required=try JSONDecoder().decode([String].self,from:Data((args["--dataflow-required-nodes"] ?? "[]").utf8))
            guard nodes.count<=128,Set(nodes).count==nodes.count,Set(nodes).isSubset(of:Set(required)) else {throw WorkflowFileLeaseError.unsafePath}
        }
        if let raw=args["--dataflow-required-nodes"] {
            let nodes=try JSONDecoder().decode([String].self,from:Data(raw.utf8));guard nodes.count<=128,Set(nodes).count==nodes.count else {throw WorkflowFileLeaseError.unsafePath}
        }
        if let raw=args["--defer-database-writes"],raw != "true" {throw WorkflowFileLeaseError.unsafePath}
        if let raw=args["--dataflow-capture-outputs"],!["true","false"].contains(raw) {throw WorkflowFileLeaseError.unsafePath}
        if let raw=args["--dataflow-measurements"],!["true","false"].contains(raw) {throw WorkflowFileLeaseError.unsafePath}
        if let raw=args["--dataflow-live-previews"],!["true","false"].contains(raw) {throw WorkflowFileLeaseError.unsafePath}
        if let raw = args["--dataflow-cache-limit-bytes"] {
            guard policy != "off", let limit = Int(raw), String(limit) == raw,
                ExecutionPreferences.preparationCacheByteBounds.contains(limit) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let raw = args["--dataflow-sampling"] {
            let decoder = JSONDecoder()
            let policy = try decoder.decode(PreparationSampling.self, from: Data(raw.utf8))
            _ = try policy.validated()
            let value = try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String:Any]
            let keys: Set<String> = policy.mode == .all ? ["schemaVersion","mode"] : policy.mode == .rowCount ? ["schemaVersion","mode","rowCount"] : ["schemaVersion","mode","percentage"]
            guard let value, Set(value.keys) == keys else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let raw = args["--dataflow-complete-nodes"] {
            let nodes = try JSONDecoder().decode([String].self, from: Data(raw.utf8))
            guard nodes.count <= 128, Set(nodes).count == nodes.count else { throw WorkflowFileLeaseError.unsafePath }
        }
        let paths = Set(files.map(\.path)), controls: Set<String> = ["input/graph.json", "input/bindings.json", "input/node-labels.json"]
        guard controls.isSubset(of: paths) else { throw WorkflowFileLeaseError.unsafePath }
        if let destination = args["--destination-bindings"] {
            guard destination == root.appendingPathComponent("input/destinations.json").path,
                  paths.contains("input/destinations.json") else {throw WorkflowFileLeaseError.unsafePath}
        } else if paths.contains("input/destinations.json") {throw WorkflowFileLeaseError.unsafePath}
        var total = 0, decisionBytes = 0, decisionWeights = 0, sourceCount = 0, reuseCount = 0, reuseBytes = 0
        for file in files {
            guard file.bytes >= 0, file.bytes <= (try fileLimit(path: file.path, workload: workload)), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
            guard file.bytes <= WorkflowPreviewCapture.Purpose.preparation.captureBytes(workload) - total else {throw WorkflowFileLeaseError.unsafePath}
            total += file.bytes
            if file.path.hasPrefix("input/decision-models/") {decisionBytes += file.bytes;if file.path.hasSuffix("/model.safetensors") {decisionWeights += 1}}
            if file.path.hasPrefix("input/sources/") {
                sourceCount += 1
                guard file.path.split(separator: "/")[2] == file.sha256 else { throw WorkflowFileLeaseError.unsafePath }
            }
            if file.path.hasPrefix("input/captured-tables/") {sourceCount+=1}
            if file.path.hasPrefix("input/file-collections/"),file.path.hasSuffix("/manifest.json") {
                sourceCount+=1;guard file.path.split(separator:"/")[2] == file.sha256 else {throw WorkflowFileLeaseError.unsafePath}
            }
            if file.path.hasPrefix("input/reuse/"), file.path != "input/reuse/manifest.json" { reuseCount += 1; reuseBytes += file.bytes }
        }
        guard total-decisionBytes <= Int(workload?.limit(.retainedBytes) ?? 200_000_000) + 1_300_000_000, decisionBytes <= 2_100_000_000, decisionWeights <= 1, sourceCount <= 128, reuseCount <= (workload.map { PreparationArtifactPolicy.inventoryCount($0) } ?? 300), reuseBytes <= Int(workload?.limit(.retainedBytes) ?? 200_000_000) else { throw WorkflowFileLeaseError.unsafePath }
        if policy == "reuse" {
            guard args["--dataflow-reuse-manifest"] == root.appendingPathComponent("input/reuse/manifest.json").path,
                  let pin = args["--dataflow-reuse-manifest-sha256"], WorkflowArchiveCodec.isHash(pin),
                  files.first(where: { $0.path == "input/reuse/manifest.json" })?.sha256 == pin, reuseCount > 0 else { throw WorkflowFileLeaseError.unsafePath }
        } else {
            guard args["--dataflow-reuse-manifest"] == nil, args["--dataflow-reuse-manifest-sha256"] == nil,
                  !paths.contains(where: { $0.hasPrefix("input/reuse/") }) else { throw WorkflowFileLeaseError.unsafePath }
        }
    }
    static func verifyInputs(arguments: [String], files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        try validate(arguments: arguments, files: files, directory: directory, workload: workload)
        for file in files { try WorkflowPreviewToolContract.verifyFile(file: file, directory: directory, limit: fileLimit(path: file.path, workload: workload)) }
        let graphBytes = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"), name: "graph.json", limit: 2_000_000)
        let graph = try DataFlowCodec.decodeGraph(graphBytes)
        guard !graph.nodes.isEmpty else { throw WorkflowFileLeaseError.unsafePath }
        let bindingsBytes = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"), name: "bindings.json", limit: 1_000_000)
        let labelsBytes = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"), name: "node-labels.json", limit: 65_536)
        let bindings = try JSONDecoder().decode([String: String].self, from: bindingsBytes)
        let labels = try JSONDecoder().decode([String: String].self, from: labelsBytes)
        let packFiles=files.filter {$0.path.hasPrefix("input/ocr-pack/")}
        if graph.nodes.contains(where:{$0.operatorID=="kwip.data.extractText"}) {
            guard Set(packFiles.map(\.path))==Set(OCRPackCatalog.assets.map {"input/ocr-pack/"+$0.name}),packFiles.allSatisfy({file in guard let asset=OCRPackCatalog.asset(String(file.path.dropFirst(15))) else {return false};return asset.sha256==file.sha256 && asset.bytes==file.bytes}) else {throw WorkflowArchiveError("Capture the complete verified English OCR pack before running.")}
        } else if !packFiles.isEmpty {throw WorkflowFileLeaseError.unsafePath}
        try verifyFrozenPreparation(files:files,directory:directory,nodes:Set(graph.nodes.filter {$0.operatorID=="kwip.data.preprocess" && $0.config["mode"] == .string("frozen")}.map(\.id)))
        try verifyDecisions(files:files,directory:directory,nodes:Set(graph.nodes.filter {$0.operatorID=="kwip.data.decide"}.map(\.id)))
        try verifyAudio(files:files,directory:directory,nodes:Set(graph.nodes.filter {$0.operatorID=="kwip.data.classifyAudio"}.map(\.id)))
        try verifySpeech(files:files,directory:directory,nodes:Set(graph.nodes.filter {$0.operatorID=="kwip.data.transcribeAudio"}.map(\.id)))
        let sourceIDs=Set(graph.nodes.filter {DataFlowSourceOperators.contains($0.operatorID)}.map(\.id))
        let sourceInventory=Dictionary(uniqueKeysWithValues:files.filter {$0.path.hasPrefix("input/sources/") || $0.path.hasPrefix("input/file-collections/") || $0.path.hasPrefix("input/captured-tables/")}.map {($0.path,$0)})
        var used=Set<String>(),memberCount=0
        for node in graph.nodes where sourceIDs.contains(node.id) {
            guard let binding=bindings[node.id],let relative=sourceInventory.keys.first(where:{directory.appendingPathComponent($0).path == binding}) else {throw WorkflowFileLeaseError.unsafePath}
            if node.operatorID == DataFlowSourceOperators.fileInput {
                _ = try FileInputConfiguration.validate(node.config)
                let parts=try WorkflowArchiveCodec.pathParts(relative)
                guard parts.count==4,parts[0]=="input",parts[1]=="file-collections",parts[3]=="manifest.json" else {throw WorkflowFileLeaseError.unsafePath}
                let bytes=try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent(parts.dropLast().joined(separator:"/")),name:"manifest.json",limit:2_000_000)
                guard WorkflowArchiveCodec.hash(bytes)==sourceInventory[relative]?.sha256 else {throw WorkflowFileLeaseError.unsafePath}
                let manifest=try FileInputCollectionManifest.decode(bytes)
                let selected=try FileInputConfiguration.validate(node.config)
                guard manifest.selection==selected.filter({FileInputCollectionManifest.selectionKeys.contains($0.key)}) else {throw WorkflowFileLeaseError.unsafePath}
                memberCount+=manifest.members.count
                guard memberCount<=1000 else {throw WorkflowFileLeaseError.unsafePath}
                used.formUnion(try manifest.ownedPaths(manifestPath:relative,inventory:sourceInventory))
            } else if node.operatorID==DataFlowSourceOperators.capturedTable {
                guard let pin=node.config["sha256"]?.stringValue,relative=="input/captured-tables/"+pin+".json",sourceInventory[relative]?.sha256==pin else {throw WorkflowFileLeaseError.unsafePath};used.insert(relative)
            } else {guard relative.hasPrefix("input/sources/") else {throw WorkflowFileLeaseError.unsafePath};used.insert(relative)}
        }
        guard Set(bindings.keys) == sourceIDs, used == Set(sourceInventory.keys),
              Set(labels.keys).isSubset(of: Set(graph.nodes.map(\.id))),
              labels.values.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.count <= 200 }) else { throw WorkflowFileLeaseError.unsafePath }
        if graph.nodes.contains(where:{$0.operatorID == "kwip.data.outputDelimited" && APITableSettings.isExternal($0.config)}),try options(arguments)["--defer-database-writes"] != "true" {throw WorkflowFileLeaseError.unsafePath}
        let outputIDs = Set(graph.nodes.filter {$0.operatorID == "kwip.data.outputDelimited" && !APITableSettings.isExternal($0.config)}.map(\.id))
        if !outputIDs.isEmpty {
            guard try options(arguments)["--destination-bindings"] != nil else {throw WorkflowFileLeaseError.unsafePath}
            let bytes = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"),name:"destinations.json",limit:1_000_000)
            let destinations = try JSONDecoder().decode([String:String].self,from:bytes)
            guard Set(destinations.keys) == outputIDs,
                  destinations.values.allSatisfy({$0.hasPrefix("/") && !$0.contains("\0") && $0.utf8.count <= 16_384}) else {throw WorkflowFileLeaseError.unsafePath}
        } else if try options(arguments)["--destination-bindings"] != nil {throw WorkflowFileLeaseError.unsafePath}
        if try options(arguments)["--dataflow-cache-policy"] == "reuse" {
            let data = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input/reuse"), name: "manifest.json", limit: 2_000_000)
            try WorkflowCompletedPreparationInventory.verify(data: data, files: files, prefix: "input/reuse/", workload: workload)
        }
    }
    private struct SpeechPin:Decodable {let identity:String;let files:[WorkflowArchiveFile]}
    static func verifyFrozenPreparation(files:[WorkflowArchiveFile],directory:URL,nodes:Set<String>)throws {
        let captured=files.filter{$0.path.hasPrefix("input/preparation-models/")}
        guard !nodes.isEmpty else {guard captured.isEmpty,!files.contains(where:{$0.path=="input/preparation-models.json"}) else {throw WorkflowFileLeaseError.unsafePath};return}
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["input","preparation-models.json"],limit:1_000_000)
        let mapping=try JSONDecoder().decode([String:[String:String]].self,from:bytes)
        guard Set(mapping.keys)==nodes else {throw WorkflowArchiveError("Learned preparation resources differ from the selected graph.")}
        var used=Set<String>()
        for pin in mapping.values {
            guard Set(pin.keys)==["path","sha256","preparationPath","preparationSHA256"],let model=pin["path"],let preparation=pin["preparationPath"],let sha=pin["sha256"],let preparationSHA=pin["preparationSHA256"] else {throw WorkflowFileLeaseError.unsafePath}
            let parts=try WorkflowArchiveCodec.pathParts(model)
            guard parts.count==3,parts[0]=="preparation-models",WorkflowArchiveCodec.isHash(parts[1]),parts[2]=="state.json",preparation=="preparation-models/"+parts[1]+"/preparation.json",captured.contains(where:{$0.path=="input/"+model && $0.sha256==sha}),captured.contains(where:{$0.path=="input/"+preparation && $0.sha256==preparationSHA}) else {throw WorkflowFileLeaseError.unsafePath}
            let state=try DiagnosticFileIO.read(root:directory,parts:["input"]+parts,limit:32_000_000)
            let rules=try DiagnosticFileIO.read(root:directory,parts:["input","preparation-models",parts[1],"preparation.json"],limit:PrebuiltPreparationAttachment.maximumBytes)
            _=try PrebuiltPreparationAttachment.decode(rules,modelBytes:state)
            used.formUnion(["input/"+model,"input/"+preparation])
        }
        guard used==Set(captured.map(\.path)) else {throw WorkflowFileLeaseError.unsafePath}
    }
    private static func verifySpeech(files:[WorkflowArchiveFile],directory:URL,nodes:Set<String>)throws {
        let captured=files.filter {$0.path.hasPrefix("input/speech-models/")}
        guard !nodes.isEmpty else {guard captured.isEmpty,!files.contains(where:{$0.path=="input/speech-models.json"}) else {throw WorkflowFileLeaseError.unsafePath};return}
        let raw=try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"),name:"speech-models.json",limit:1_000_000)
        let pins=try JSONDecoder().decode([String:SpeechPin].self,from:raw)
        guard Set(pins.keys)==nodes else {throw WorkflowFileLeaseError.unsafePath}
        var expected:[String:WorkflowArchiveFile]=[:]
        for pin in pins.values {
            guard WorkflowArchiveCodec.isHash(pin.identity),Set(pin.files.map(\.path)).count==pin.files.count,Set(["config.json","tokenizer.json","preprocessor_config.json"]).isSubset(of:Set(pin.files.map(\.path))),try ModelAssetFiles.sourceIdentity(pin.files)==pin.identity else {throw WorkflowFileLeaseError.unsafePath}
            for file in pin.files {guard ModelHubRequest.allowedFile(file.path),!file.path.contains("/") else {throw WorkflowFileLeaseError.unsafePath};let path="input/speech-models/"+pin.identity+"/"+file.path;expected[path] = .init(path:path,bytes:file.bytes,sha256:file.sha256)}
        }
        guard Set(expected.keys)==Set(captured.map(\.path)),captured.allSatisfy({expected[$0.path]==$0}) else {throw WorkflowFileLeaseError.unsafePath}
    }
    private struct AudioPin:Decodable {let identity:String;let files:[WorkflowArchiveFile]}
    private static func verifyDecisions(files:[WorkflowArchiveFile],directory:URL,nodes:Set<String>)throws {
        let captured=files.filter {$0.path.hasPrefix("input/decision-models/")}
        guard !nodes.isEmpty else {guard captured.isEmpty,!files.contains(where:{$0.path=="input/decision-models.json"}) else {throw WorkflowFileLeaseError.unsafePath};return}
        let raw=try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"),name:"decision-models.json",limit:1_000_000)
        let pins=try JSONDecoder().decode([String:AudioPin].self,from:raw)
        guard Set(pins.keys)==nodes else {throw WorkflowFileLeaseError.unsafePath}
        var expected:[String:WorkflowArchiveFile]=[:]
        for pin in pins.values {
            guard WorkflowArchiveCodec.isHash(pin.identity),Set(pin.files.map(\.path)).count==pin.files.count,Set(["rl_agent_config.json","encoder/config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json"]).isSubset(of:Set(pin.files.map(\.path))),try ModelAssetFiles.sourceIdentity(pin.files)==pin.identity else {throw WorkflowFileLeaseError.unsafePath}
            for file in pin.files {guard ModelHubRequest.allowedAssetFile(file.path) else {throw WorkflowFileLeaseError.unsafePath};let path="input/decision-models/"+pin.identity+"/"+file.path;expected[path] = .init(path:path,bytes:file.bytes,sha256:file.sha256)}
        }
        guard Set(expected.keys)==Set(captured.map(\.path)),captured.allSatisfy({expected[$0.path]==$0}) else {throw WorkflowFileLeaseError.unsafePath}
    }
    private static func verifyAudio(files:[WorkflowArchiveFile],directory:URL,nodes:Set<String>)throws {
        let captured=files.filter {$0.path.hasPrefix("input/audio-models/")}
        guard !nodes.isEmpty else {guard captured.isEmpty,!files.contains(where:{$0.path=="input/audio-models.json"}) else {throw WorkflowFileLeaseError.unsafePath};return}
        let raw=try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"),name:"audio-models.json",limit:1_000_000)
        let pins=try JSONDecoder().decode([String:AudioPin].self,from:raw)
        guard Set(pins.keys)==nodes else {throw WorkflowFileLeaseError.unsafePath}
        var expected:[String:WorkflowArchiveFile]=[:]
        for pin in pins.values {
            guard WorkflowArchiveCodec.isHash(pin.identity),Set(pin.files.map(\.path)).count==pin.files.count,Set(["config.json","preprocessor_config.json"]).isSubset(of:Set(pin.files.map(\.path))),try ModelAssetFiles.sourceIdentity(pin.files)==pin.identity else {throw WorkflowFileLeaseError.unsafePath}
            for file in pin.files {guard ModelHubRequest.allowedFile(file.path),!file.path.contains("/") else {throw WorkflowFileLeaseError.unsafePath};let path="input/audio-models/"+pin.identity+"/"+file.path;expected[path] = .init(path:path,bytes:file.bytes,sha256:file.sha256)}
        }
        guard Set(expected.keys)==Set(captured.map(\.path)),captured.allSatisfy({expected[$0.path]==$0}) else {throw WorkflowFileLeaseError.unsafePath}
    }

}
