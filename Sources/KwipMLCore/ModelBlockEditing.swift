import Foundation

/// Pure parent-document transactions. A copied definition reuses structure, never tied parameter state.
public enum ModelBlockEditing {
    public static func beginDraft(_ model:ModelAuthoringDocument,path:[String])throws->ModelAuthoringDocument {
        try ModelProjectCodec.validate(.blockModel(model))
        let chain=try resolve(model,path:path)
        if !model.drafts.isEmpty {_ = try currentDraft(model,path:path,expectedBaseRevision:model.revision);return model}
        var result=model
        result.drafts=[ModelBlockDraft(projectID:model.projectID,instancePath:path,baseReference:chain.last!.definition.reference,baseRevision:model.revision,proposed:chain.last!.definition)]
        try ModelProjectCodec.validate(.blockModel(result));return result
    }
    public static func updateDraft(_ model:ModelAuthoringDocument,proposed:CompositeDefinition,expectedBaseRevision:UInt64,path:[String])throws->ModelAuthoringDocument {
        _ = try currentDraft(model,path:path,expectedBaseRevision:expectedBaseRevision)
        var result=model
        if let policy=proposed.composition {
            _ = try JSONDecoder().decode(CompositionPolicy.self,from:JSONEncoder().encode(policy))
            let ids=proposed.body.nodes.map(\.id)+proposed.body.instances.map(\.id)
            guard Set(ids).count==ids.count,Set(policy.children.map(\.localID))==Set(ids) else {throw failure("Refresh composition membership explicitly before saving the selected policy.")}
            if result.schemaVersion==1 {
                result=try ModelProjectCodec.enablingComposition(result)
                result.drafts[0].migrationOrigin = .model(modelSchemaVersion:model.schemaVersion,sourceSchemaVersion:model.source.schemaVersion)
            }
        }
        result.drafts[0].proposed=proposed
        try ModelProjectCodec.validate(.blockModel(result));return result
    }
    public static func discardDraft(_ model:ModelAuthoringDocument,path:[String],expectedBaseRevision:UInt64)throws->ModelAuthoringDocument {
        _ = try currentDraft(model,path:path,expectedBaseRevision:expectedBaseRevision)
        let origin=model.drafts[0].migrationOrigin
        var result=model;result.drafts=[]
        if case .model(let version,let sourceVersion)=origin {result.schemaVersion=version;result.source.schemaVersion=sourceVersion}
        try ModelProjectCodec.validate(.blockModel(result));return result
    }
    public static func applyDraft(_ model:ModelAuthoringDocument,path:[String],expectedBaseRevision:UInt64,registry:ComponentRegistry?=nil,definitionValidator:((CompositeDefinition,[CompositeDefinition],ComponentRegistry)throws->Void)?=nil,idProvider:()->String={UUID().uuidString})throws->ModelAuthoringDocument {
        let draft=try currentDraft(model,path:path,expectedBaseRevision:expectedBaseRevision),chain=try resolve(model,path:path)
        guard model.revision<UInt64.max else {throw failure("Parent revision is exhausted.")}
        guard model.source.definitions.count+chain.count<=64 else {throw failure("Applying this Block would exceed 64 retained definitions. No revisions were pruned or changed.")}
        let registry=try registry ?? ComponentRegistry.builtIn()
        var result=model,used=Set(model.source.definitions.map{$0.reference.id})
        func freshReference()throws->CompositeDefinitionReference {
            let id=idProvider()
            guard !id.isEmpty,id.unicodeScalars.count<=128,used.insert(id).inserted else {throw failure("A private Block copy needs a fresh bounded definition identity.")}
            return CompositeDefinitionReference(id)
        }
        var edited=draft.proposed;edited.reference=try freshReference()
        if let definitionValidator {try definitionValidator(edited,result.source.definitions+[edited],registry)}
        else {try CompositeDefinitionValidation.validate(edited,definitions:result.source.definitions+[edited],registry:registry)}
        result.source.definitions.append(edited)
        var replacement=edited.reference
        if chain.count>1 {
            for depth in stride(from:chain.count-2,through:0,by:-1) {
                var ancestor=chain[depth].definition
                guard let index=ancestor.body.instances.firstIndex(where:{$0.id == path[depth+1]}) else {throw failure("The selected child no longer belongs to its ancestor.")}
                ancestor.body.instances[index].definition=replacement
                ancestor.reference=try freshReference()
                if let definitionValidator {try definitionValidator(ancestor,result.source.definitions+[ancestor],registry)}
                else {try CompositeDefinitionValidation.validate(ancestor,definitions:result.source.definitions+[ancestor],registry:registry)}
                result.source.definitions.append(ancestor);replacement=ancestor.reference
            }
        }
        guard let rootIndex=result.source.root.instances.firstIndex(where:{$0.id == path[0]}),let replacementDefinition=result.source.definitions.first(where:{$0.reference == replacement}) else {throw failure("Root Block path changed.")}
        // Root may be unfinished, but replacing an interface cannot strand existing wires or overrides.
        let rootInstance=result.source.root.instances[rootIndex]
        let fields=replacementDefinition.parameters.map(\.field)
        guard Set(rootInstance.overrides.keys).isSubset(of:Set(fields.map(\.key))),fields.allSatisfy({$0.accepts(rootInstance.overrides[$0.key] ?? $0.defaultValue)}) else {throw failure("Existing instance overrides are incompatible with the edited Block interface.")}
        for edge in result.source.root.edges {
            if edge.source == rootInstance.id && !replacementDefinition.outputs.contains(where:{$0.id == edge.sourcePort}) {throw failure("An existing parent wire targets a removed Block output.")}
            if edge.target == rootInstance.id && !replacementDefinition.inputs.contains(where:{$0.id == edge.targetPort}) {throw failure("An existing parent wire targets a removed Block input.")}
        }
        result.source.root.instances[rootIndex].definition=replacement
        result.revision+=1;result.drafts=[]
        try ModelProjectCodec.validate(.blockModel(result));return result
    }
    private struct PathItem {let instance:CompositeInstance;let definition:CompositeDefinition}
    private static func resolve(_ model:ModelAuthoringDocument,path:[String])throws->[PathItem] {
        guard (1...8).contains(path.count) else {throw failure("Block path must contain 1–8 instance identities.")}
        var scope=model.source.root,chain:[PathItem]=[]
        for id in path {
            guard let instance=scope.instances.first(where:{$0.id == id}),let definition=model.source.definitions.first(where:{$0.reference == instance.definition}) else {throw failure("Block instance path no longer resolves.")}
            chain.append(PathItem(instance:instance,definition:definition));scope=definition.body
        }
        return chain
    }
    private static func currentDraft(_ model:ModelAuthoringDocument,path:[String],expectedBaseRevision:UInt64)throws->ModelBlockDraft {
        try ModelProjectCodec.validate(.blockModel(model))
        guard model.revision == expectedBaseRevision,let draft=model.drafts.first,draft.projectID == model.projectID,draft.instancePath == path,draft.baseRevision == expectedBaseRevision,try resolve(model,path:path).last?.definition.reference == draft.baseReference else {throw failure("Child edit context is stale or belongs to a different parent/path.")}
        return draft
    }
    private static func failure(_ message:String)->CompositeContractError {CompositeContractError(.immutableRevision,message)}
}
