import Foundation
import KwipMLCore

/// Standalone training accepts immutable owned data, optional continuation and composite sources.
enum WorkflowModelTrainingToolContract {
    static func fileLimit(path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") }), parts.first == "input" else { throw WorkflowFileLeaseError.unsafePath }
        switch path {
        case "input/graph.json", "input/preparation/manifest.json", "input/composite/source.json", "input/model-registry.json": return 2_000_000
        case "input/continuation-parent/manifest.json": return 20_000_000
        case "input/continuation-parent/checkpoint.pt": return 100_000_000
        case "input/continuation-parent/metrics.json", "input/continuation-parent/rate-trace.json": return 4_000_000
        default:
            guard parts.count == 3 else { throw WorkflowFileLeaseError.unsafePath }
            if parts[1] == "data" { return 10_000_000 }
            if parts[1] == "preparation" { return workload != nil && PreparationArtifactPolicy.isCapacityFileName(parts[2]) ? Int(workload!.limit(.retainedBytes)) : 50_000_000 }
            throw WorkflowFileLeaseError.unsafePath
        }
    }
    private static func options(_ arguments: [String]) throws -> [String: String] {
        let allowed: Set<String> = ["--graph", "--output", "--device", "--seed", "--epochs", "--batch-size", "--learning-rate",
            "--dataset-csv", "--sample-limit", "--split-seed", "--validation-percent", "--prepared-dataflow", "--prepared-manifest-sha256",
            "--prepared-graph-digest", "--prepared-training-node", "--prepared-training-port", "--prepared-validation-node", "--prepared-validation-port",
            "--feature-columns-json", "--target-column", "--resume-checkpoint", "--finetune-checkpoint", "--composite-source", "--composite-source-sha256",
            "--training-control-json","--training-progress-json","--sample-capture-json", "--initialization-recipe-json", "--optimizer-recipe-json", "--scheduler-recipe-json", "--capture-gradients", "--gradient-every-epochs", "--gradient-max-batches-per-epoch", "--gradient-max-captures"]
        guard arguments.count <= 64, arguments.allSatisfy({ !$0.isEmpty && !$0.contains("\0") }) else { throw WorkflowFileLeaseError.unsafePath }
        var result: [String: String] = [:], index = 0
        while index < arguments.count {
            let key = arguments[index]
            guard allowed.contains(key), result[key] == nil else { throw WorkflowFileLeaseError.unsafePath }
            if key == "--capture-gradients" { result[key] = "true"; index += 1 }
            else {
                guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else { throw WorkflowFileLeaseError.unsafePath }
                guard arguments[index + 1].utf8.count <= (key == "--initialization-recipe-json" ? InitializationConfiguration.maximumRecipeBytes : 8192) else { throw WorkflowFileLeaseError.unsafePath }
                result[key] = arguments[index + 1]; index += 2
            }
        }
        return result
    }
    private static func integer(_ value: String?, in bounds: ClosedRange<Int>) -> Bool {
        guard let value, let number = Int(value) else { return false }
        return String(number) == value && bounds.contains(number)
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?, workload: ExecutionWorkloadPolicy? = nil) throws {
        let args = try options(arguments)
        guard let graph = args["--graph"], graph.hasPrefix("/"), graph.hasSuffix("/input/graph.json"),
              !graph.contains("//"), !graph.split(separator: "/").contains("."), !graph.split(separator: "/").contains("..") else { throw WorkflowFileLeaseError.unsafePath }
        let root = directory ?? URL(fileURLWithPath: graph).deletingLastPathComponent().deletingLastPathComponent()
        guard root.isFileURL, UUID(uuidString: root.lastPathComponent)?.uuidString == root.lastPathComponent,
              UUID(uuidString: root.deletingLastPathComponent().lastPathComponent)?.uuidString == root.deletingLastPathComponent().lastPathComponent,
              root.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations",
              graph == root.appendingPathComponent("input/graph.json").path, args["--output"] == root.appendingPathComponent("artifacts").path,
              ["auto", "cpu", "mps"].contains(args["--device"] ?? ""),
              integer(args["--seed"], in: 0...Int(UInt32.max)), integer(args["--epochs"], in: 1...10000), integer(args["--batch-size"], in: 1...100000),
              let rate = args["--learning-rate"].flatMap(Double.init), rate.isFinite, (args["--optimizer-recipe-json"] != nil ? (0...OptimizerConfiguration.scalarMaximum).contains(rate) : (rate>0 && rate<=1)),
              (1...WorkflowPreviewCapture.Purpose.modelTraining.captureCount(workload)).contains(files.count), Set(files.map(\.path)).count == files.count else { throw WorkflowFileLeaseError.unsafePath }
        if let raw = args["--initialization-recipe-json"] {
            let value = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
            let configuration = try InitializationConfiguration.decodeRecipe(value)
            guard try WorkflowPhaseArtifacts.encode(value) == WorkflowPhaseArtifacts.encode(configuration.recipeValue) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let raw=args["--optimizer-recipe-json"] {
            guard raw.utf8.count<=8192 else {throw WorkflowFileLeaseError.unsafePath}
            _ = try OptimizerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)))
        }
        if let raw=args["--scheduler-recipe-json"] {
            guard raw.utf8.count<=8192,args["--optimizer-recipe-json"] != nil else {throw WorkflowFileLeaseError.unsafePath}
            _ = try SchedulerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)))
        }
        var total = 0
        for file in files {
            guard file.bytes >= 0, file.bytes <= (try fileLimit(path: file.path, workload: workload)), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
            total += file.bytes
            guard total <= WorkflowPreviewCapture.Purpose.modelTraining.captureBytes(workload) else { throw WorkflowFileLeaseError.unsafePath }
        }
        var expected: Set<String> = ["input/graph.json"]
        try WorkflowModelDataSelection.validate(args, files: files, root: root, expected: &expected, workload: workload)
        if let raw = args["--feature-columns-json"] {
            guard let names = try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String],
                  names.count == 2, names.allSatisfy({ $0.unicodeScalars.count <= 200 }) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let raw = args["--training-control-json"] { _ = try JSONDecoder().decode(TrainingControlPolicy.self,from:Data(raw.utf8)) }
        if let raw = args["--training-progress-json"] { _ = try TrainingReportingPolicy.decode(Data(raw.utf8)) }
        if let raw = args["--sample-capture-json"] {
            let bytes = Data(raw.utf8)
            guard bytes.count <= 1024 else { throw WorkflowFileLeaseError.unsafePath }
            _ = try JSONDecoder().decode(TrainingSampleCapturePolicy.self, from: bytes)
        }
        let gradients: Set<String> = ["--gradient-every-epochs", "--gradient-max-batches-per-epoch", "--gradient-max-captures"]
        if args["--capture-gradients"] != nil {
            guard integer(args["--gradient-every-epochs"], in: GradientCaptureConfiguration.epochCadenceBounds),
                  integer(args["--gradient-max-batches-per-epoch"], in: GradientCaptureConfiguration.batchBounds),
                  integer(args["--gradient-max-captures"], in: GradientCaptureConfiguration.captureBounds) else { throw WorkflowFileLeaseError.unsafePath }
        } else if !gradients.isDisjoint(with: args.keys) { throw WorkflowFileLeaseError.unsafePath }
        let parents = ["--resume-checkpoint", "--finetune-checkpoint"].filter { args[$0] != nil }
        guard parents.count <= 1 else { throw WorkflowFileLeaseError.unsafePath }
        if let parent = parents.first {
            guard args[parent] == root.appendingPathComponent("input/continuation-parent/checkpoint.pt").path else { throw WorkflowFileLeaseError.unsafePath }
            expected.formUnion(["input/continuation-parent/checkpoint.pt", "input/continuation-parent/manifest.json", "input/continuation-parent/metrics.json"])
            if files.contains(where:{$0.path == "input/continuation-parent/rate-trace.json"}) {expected.insert("input/continuation-parent/rate-trace.json")}
        }
        if args["--composite-source"] != nil || args["--composite-source-sha256"] != nil {
            guard args["--composite-source"] == root.appendingPathComponent("input/composite/source.json").path,
                  WorkflowArchiveCodec.isHash(args["--composite-source-sha256"] ?? ""),
                  files.first(where: { $0.path == "input/composite/source.json" })?.sha256 == args["--composite-source-sha256"] else { throw WorkflowFileLeaseError.unsafePath }
            expected.insert("input/composite/source.json")
            if files.contains(where: { $0.path == "input/model-registry.json" }) { expected.insert("input/model-registry.json") }
        }
        guard Set(files.map(\.path)) == expected else { throw WorkflowFileLeaseError.unsafePath }
    }
    static func verifyInputs(arguments: [String], files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        try validate(arguments: arguments, files: files, directory: directory, workload: workload)
        for file in files { try WorkflowPreviewToolContract.verifyFile(file: file, directory: directory, limit: fileLimit(path: file.path, workload: workload)) }
        let args = try options(arguments)
        if args["--composite-source"] != nil {
            let bytes = try DiagnosticFileIO.read(root: directory, parts: ["input","composite","source.json"], limit: 2_000_000)
            let source = try WorkflowArchiveCodec.decodeCompositeSource(bytes)
            let selected = source.nativeExpansionContractVersion == 2
            guard selected == files.contains(where: { $0.path == "input/model-registry.json" }) else { throw WorkflowFileLeaseError.unsafePath }
            if selected {
                let registryBytes = try DiagnosticFileIO.read(root: directory, parts: ["input","model-registry.json"], limit: 2_000_000)
                let registry = try JSONDecoder().decode(ComponentRegistry.self, from: registryBytes)
                let graphBytes = try DiagnosticFileIO.read(root: directory, parts: ["input","graph.json"], limit: 2_000_000)
                let graph = try JSONDecoder().decode(GraphDocument.self, from: graphBytes)
                let expansion = try source.verifiedExpansion(registry: registry)
                guard try GraphSemantics.canonicalData(expansion.graph, registry: registry) == GraphSemantics.canonicalData(graph, registry: registry) else { throw WorkflowFileLeaseError.unsafePath }
            }
        }
        if args["--resume-checkpoint"] != nil || args["--finetune-checkpoint"] != nil {
            let parentFiles=files.filter {$0.path.hasPrefix("input/continuation-parent/")}
            var contents:[String:Data]=[:]
            for file in parentFiles {
                let name=String(file.path.dropFirst("input/continuation-parent/".count))
                contents[name]=try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input/continuation-parent"),name:name,limit:fileLimit(path:file.path))
            }
            try WorkflowCheckpointInventory.verify(contents)
        }
        if args["--prepared-dataflow"] != nil {
            let bytes = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input/preparation"), name: "manifest.json", limit: 2_000_000)
            guard WorkflowArchiveCodec.hash(bytes) == args["--prepared-manifest-sha256"] else { throw WorkflowFileLeaseError.unsafePath }
            try WorkflowCompletedPreparationInventory.verify(data: bytes, files: files, prefix: "input/preparation/", expectedGraphDigest: args["--prepared-graph-digest"], workload: workload)
            guard let training = args["--prepared-training-node"], let validation = args["--prepared-validation-node"] else { throw WorkflowFileLeaseError.unsafePath }
            _ = try WorkflowCompletedPreparationInventory.verifyCoverage(data: bytes, files: files, prefix: "input/preparation/", directory: directory, selectedNodes: [training, validation])
        }
    }
}

/// Shared data choice rules; preview and training apply their own model-mode controls.
enum WorkflowModelDataSelection {
    static func validate(_ args: [String: String], files: [WorkflowArchiveFile], root: URL, expected: inout Set<String>, workload: ExecutionWorkloadPolicy? = nil) throws {
        let preparation: Set<String> = ["--prepared-dataflow", "--prepared-manifest-sha256", "--prepared-graph-digest", "--prepared-training-node", "--prepared-training-port", "--prepared-validation-node", "--prepared-validation-port"]
        let sampling: Set<String> = ["--sample-limit", "--split-seed", "--validation-percent"]
        let keys = Set(args.keys)
        if args["--prepared-dataflow"] != nil {
            guard preparation.isSubset(of: keys), sampling.isDisjoint(with: keys), args["--dataset-csv"] == nil,
                  args["--prepared-dataflow"] == root.appendingPathComponent("input/preparation/manifest.json").path,
                  WorkflowArchiveCodec.isHash(args["--prepared-manifest-sha256"] ?? ""), WorkflowArchiveCodec.isHash(args["--prepared-graph-digest"] ?? ""),
                  files.first(where: { $0.path == "input/preparation/manifest.json" })?.sha256 == args["--prepared-manifest-sha256"] else { throw WorkflowFileLeaseError.unsafePath }
            let prepared = files.filter { $0.path.hasPrefix("input/preparation/") && $0.path != "input/preparation/manifest.json" }
            guard (1...(workload.map { PreparationArtifactPolicy.inventoryCount($0) } ?? 300)).contains(prepared.count), prepared.reduce(0, { $0 + $1.bytes }) <= Int(workload?.limit(.retainedBytes) ?? 200_000_000) else { throw WorkflowFileLeaseError.unsafePath }
            expected.insert("input/preparation/manifest.json"); expected.formUnion(prepared.map(\.path))
            try validateColumns(args, csv: false)
        } else {
            guard preparation.isDisjoint(with: keys), sampling.isSubset(of: keys),
                  let split = args["--split-seed"].flatMap(UInt32.init), String(split) == args["--split-seed"],
                  let sample = args["--sample-limit"].flatMap(Int.init), String(sample) == args["--sample-limit"], sample == 0 || (20...100_000).contains(sample),
                  let validation = args["--validation-percent"].flatMap(Int.init), String(validation) == args["--validation-percent"], (5...50).contains(validation) else { throw WorkflowFileLeaseError.unsafePath }
            if let csv = args["--dataset-csv"] {
                let name = URL(fileURLWithPath: csv).lastPathComponent, path = "input/data/" + name
                guard csv == root.appendingPathComponent(path).path else { throw WorkflowFileLeaseError.unsafePath }
                _ = try WorkflowPreviewToolContract.fileLimit(path: path); expected.insert(path)
                try validateColumns(args, csv: true)
            } else {
                guard args["--feature-columns-json"] == nil, args["--target-column"] == nil else { throw WorkflowFileLeaseError.unsafePath }
            }
        }
    }
    private static func validateColumns(_ args: [String: String], csv: Bool) throws {
        guard let raw = args["--feature-columns-json"], let names = try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String],
              (1...128).contains(names.count), !csv || names.count == 2,
              Set(names).count == names.count,
              names.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && (csv || $0.count <= 200) }),
              let target = args["--target-column"], !target.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              (csv || target.count <= 200), !names.contains(target) else { throw WorkflowFileLeaseError.unsafePath }
    }
}
