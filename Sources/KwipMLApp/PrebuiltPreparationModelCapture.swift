import Foundation
import KwipMLCore

/// Captures only the inert state and its model-bound preparation through Load model.
/// The graph itself has no filesystem paths or hidden resource grants.
enum PrebuiltPreparationModelCapture {
    @MainActor static func inputs(document:WorkflowDocument,store:ModelLoaderStore,nodeIDs:Set<String>?=nil)async throws->[WorkflowPreviewCapture.Input] {
        var selected:[String:ModelAssetReference]=[:]
        for node in document.nodes {
            guard case .dataOperator(let id,let config)=node.payload,id=="kwip.data.preprocess",config["mode"] == .string("frozen"),nodeIDs == nil || nodeIDs!.contains(node.address.localID) else {continue}
            let edges=document.edges.filter {$0.to.address==node.address && $0.to.port=="model"}
            guard edges.count==1,edges[0].kind == .frozenModel,let source=document.nodes.first(where:{$0.address==edges[0].from.address}),case .frozenModel=source.payload,let ref=store.reference(source.address),ref.hasValidPins,ref.metadata["task"] == .string("prebuilt"),Set(ref.files.map(\.path))==["state.json","preparation.json"] else {throw WorkflowArchiveError("Connect Load model with a Model + learned preparation package to Apply saved rules.")}
            selected[node.address.localID]=ref
        }
        guard !selected.isEmpty else{return []}
        let root=store.root
        let worker=Task.detached(priority:.utility) {
            var result:[WorkflowPreviewCapture.Input]=[],manifest:[String:JSONValue]=[:],seen=Set<String>()
            for (node,ref) in selected.sorted(by:{$0.key<$1.key}) {
                let folder=ref.directory(root:root)
                guard let state=ref.files.first(where:{$0.path=="state.json"}),let preparation=ref.files.first(where:{$0.path=="preparation.json"}) else {throw WorkflowFileLeaseError.unsafePath}
                let prefix="preparation-models/"+ref.identity+"/"
                manifest[node] = .object(["path":.string(prefix+"state.json"),"sha256":.string(state.sha256),"preparationPath":.string(prefix+"preparation.json"),"preparationSHA256":.string(preparation.sha256)])
                if seen.insert(ref.identity).inserted {
                    let stateBytes=try DiagnosticFileIO.read(root:folder,parts:["state.json"],limit:32_000_000)
                    let preparationBytes=try DiagnosticFileIO.read(root:folder,parts:["preparation.json"],limit:PrebuiltPreparationAttachment.maximumBytes)
                    guard WorkflowArchiveCodec.hash(stateBytes)==state.sha256,WorkflowArchiveCodec.hash(preparationBytes)==preparation.sha256 else {throw WorkflowArchiveError("Loaded learned preparation changed.")}
                    _=try PrebuiltPreparationAttachment.decode(preparationBytes,modelBytes:stateBytes)
                    for file in [state,preparation] {
                        try Task.checkCancellation()
                        result.append(.file(path:"input/"+prefix+file.path,source:folder.appendingPathComponent(file.path),expectedSHA256:file.sha256))
                    }
                }
            }
            result.append(.bytes(path:"input/preparation-models.json",data:try WorkflowArchiveCodec.encode(manifest)))
            return result
        }
        return try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
    }
}
