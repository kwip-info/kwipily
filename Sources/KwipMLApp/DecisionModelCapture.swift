import Foundation
import KwipMLCore

/// The real Load model edge grants only an immutable, verified resource snapshot.
/// Graph configuration never carries a filesystem path or credential.
enum DecisionModelCapture {
    @MainActor static func inputs(document:WorkflowDocument,store:ModelLoaderStore)async throws->[WorkflowPreviewCapture.Input] {
        var selected:[String:ModelAssetReference]=[:]
        for node in document.nodes {
            guard case .dataOperator(let id,_)=node.payload,id=="kwip.data.decide" else {continue}
            let edges=document.edges.filter {$0.to.address==node.address && $0.to.port=="model"}
            guard edges.count==1,edges[0].kind == .frozenModel,let reference=store.reference(edges[0].from.address),reference.hasValidPins,reference.metadata["task"] == .string("typedDecisions"),reference.metadata["architecture"] == .string("laya-modernbert") else {throw WorkflowArchiveError("Connect a verified typed-decision Load model or completed Train model to Decide. Text generators cannot score typed questions.")}
            selected[node.address.localID]=reference
        }
        guard !selected.isEmpty else {return []}
        let root=store.root
        let worker=Task.detached(priority:.utility) {
            var result:[WorkflowPreviewCapture.Input]=[],manifest:[String:JSONValue]=[:],seen=Set<String>()
            for (node,ref) in selected.sorted(by:{$0.key<$1.key}) {
                let folder=ref.directory(root:root)
                manifest[node] = .object(["identity":.string(ref.identity),"files":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(ref.files))])
                if seen.insert(ref.identity).inserted {
                    for file in ref.files {
                        try Task.checkCancellation()
                        try WorkflowPreviewToolContract.verifyFile(file:file,directory:folder,limit:file.bytes)
                        result.append(.file(path:"input/decision-models/"+ref.identity+"/"+file.path,source:folder.appendingPathComponent(file.path),expectedSHA256:file.sha256))
                    }
                }
            }
            result.append(.bytes(path:"input/decision-models.json",data:try WorkflowArchiveCodec.encode(manifest)))
            return result
        }
        return try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
    }
}
