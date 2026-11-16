import Foundation

/// Protected source-only editing; complete execution always uses the caller's captured registry.
public enum ModuleCompositionLearningDraft {
    public static func model(projectID:String,task:ModuleCompositionLearningTask,prepared:Bool=false)throws->ModelAuthoringDocument {
        let source=try prepared ? ModuleCompositionLearningExamples.preparedSource(task:task):ModuleCompositionLearningExamples.defaultSource(task:task)
        let value=ModelAuthoringDocument(projectID:projectID,source:source,schemaVersion:2)
        try validate(model:value,task:task,projectID:projectID,complete:false)
        return value
    }
    public static func validate(model:ModelAuthoringDocument,task:ModuleCompositionLearningTask,projectID:String,complete:Bool,registry:ComponentRegistry?=nil)throws {
        guard model.schemaVersion==2,model.kind=="modelProject",model.projectID==projectID else {throw failure("Keep this exercise's current model identity.")}
        try ModelProjectCodec.validate(.blockModel(model))
        try validateSource(model.source,task:task)
        for draft in model.drafts {
            guard draft.migrationOrigin==nil else {throw failure("This exercise already uses the current composition version.")}
            var candidate=model.source
            guard let index=candidate.definitions.firstIndex(where:{$0.reference==draft.baseReference}) else {throw failure("The selected block revision no longer exists.")}
            candidate.definitions[index]=draft.proposed
            try validateSource(candidate,task:task)
        }
        if complete {
            guard model.drafts.isEmpty,let registry else {throw failure("Apply or discard the block draft and select the verified lesson runtime before checking.")}
            _ = try CompositeExpander.expand(model.source,registry:registry,context:.explicitObjective)
        }
    }
    public static func definitionValidator(task:ModuleCompositionLearningTask)->(CompositeDefinition,[CompositeDefinition],ComponentRegistry)throws->Void {
        { definition,definitions,_ in
            let template=try ModuleCompositionLearningExamples.preparedSource(task:task)
            var values=definitions
            if let index=values.firstIndex(where:{$0.reference==definition.reference}) {values[index]=definition} else {values.append(definition)}
            let matcher=try RoleMatcher(source:CompositeProjectDocument(schemaVersion:3,root:template.root,definitions:values),template:template)
            guard template.definitions.contains(where:{(try? matcher.definition(definition,expected:$0,active:[definition.reference])) != nil}) else {throw failure("Keep supplied layer membership and settings while repairing connections and container choices.")}
        }
    }
    public static func submission(model:ModelAuthoringDocument,task:ModuleCompositionLearningTask,registry:ComponentRegistry)throws->ModuleCompositionLearningSubmission {
        try validate(model:model,task:task,projectID:model.projectID,complete:true,registry:registry)
        return try ModuleCompositionLearningSubmission(source:model.source,task:task,registry:registry)
    }
    static func validateSource(_ source:CompositeProjectDocument,task:ModuleCompositionLearningTask)throws {
        let template=try ModuleCompositionLearningExamples.preparedSource(task:task)
        guard source.schemaVersion==3,source.execution==nil,(1...64).contains(source.definitions.count) else {throw failure("Keep the supplied source version, private definitions, and checker-owned execution settings.")}
        // Structural storage validation is independent of any live registry or runtime claim.
        try ModelProjectCodec.validate(.blockModel(.init(projectID:"DA0A0667-64F2-487E-9C0A-91F58CB7AFDA",source:source,schemaVersion:2)))
        let matcher=try RoleMatcher(source:source,template:template)
        try matcher.body(source.root,expected:template.root,active:[])
        for definition in source.definitions {
            guard template.definitions.contains(where:{(try? matcher.definition(definition,expected:$0,active:[definition.reference])) != nil}) else {throw failure("Every retained revision must preserve one supplied block role.")}
        }
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        guard try encoder.encode(source).count<=2_000_000 else {throw failure("Composition source exceeds its unchanged byte budget.")}
    }
    private struct RoleMatcher {
        let definitions:[CompositeDefinitionReference:CompositeDefinition]
        let originals:[CompositeDefinitionReference:CompositeDefinition]
        init(source:CompositeProjectDocument,template:CompositeProjectDocument)throws {
            var values:[CompositeDefinitionReference:CompositeDefinition]=[:]
            for value in source.definitions {
                guard values.updateValue(value,forKey:value.reference)==nil else {throw failure("Private definition references must resolve uniquely.")}
            }
            definitions=values;originals=Dictionary(uniqueKeysWithValues:template.definitions.map{($0.reference,$0)})
        }
        func body(_ value:CompositeGraph,expected:CompositeGraph,active:Set<CompositeDefinitionReference>)throws {
            guard value.configurationVersion==2,value.nodes.count==expected.nodes.count,value.instances.count==expected.instances.count,value.edges.count<=32 else {throw failure("Keep supplied block membership and bounded connections.")}
            for (node,fixed) in zip(value.nodes,expected.nodes) {
                guard node.id==fixed.id,node.type==fixed.type,node.config==fixed.config,node.x.isFinite,node.y.isFinite else {throw failure("Keep the supplied layer settings; edit connections and container structure.")}
            }
            for (instance,fixed) in zip(value.instances,expected.instances) {
                guard instance.id==fixed.id,instance.overrides==fixed.overrides,instance.x.isFinite,instance.y.isFinite,
                      let child=definitions[instance.definition],let original=originals[fixed.definition],!active.contains(instance.definition),active.count<3 else {throw failure("Keep each private instance role, with resolved nonrecursive references.")}
                try definition(child,expected:original,active:active.union([instance.definition]))
            }
            let ids=Set(value.nodes.map(\.id)+value.instances.map(\.id))
            guard Set(value.edges.map(\.id)).count==value.edges.count,value.edges.allSatisfy({ids.contains($0.source) && ids.contains($0.target)}) else {throw failure("Connections must belong to the supplied block.")}
        }
        func definition(_ value:CompositeDefinition,expected:CompositeDefinition,active:Set<CompositeDefinitionReference>)throws {
            guard value.parameters==expected.parameters,value.integerProducts==nil,let policy=value.composition,value.inputs.count<=4,value.outputs.count<=4 else {throw failure("Keep supplied fixed controls and explicit bounded container interfaces.")}
            try body(value.body,expected:expected.body,active:active)
            let ids=Set(value.body.nodes.map(\.id)+value.body.instances.map(\.id))
            let encoder=JSONEncoder()
            _ = try JSONDecoder().decode(CompositionPolicy.self,from:encoder.encode(policy))
            guard Set(policy.children.map(\.localID))==ids,policy.children.count==ids.count,
                  value.inputs.flatMap(\.targets).allSatisfy({ids.contains($0.componentID)}),value.outputs.allSatisfy({ids.contains($0.source.componentID)}) else {throw failure("Store every supplied child once and use boundaries belonging to this block.")}
        }
    }
    private static func failure(_ message:String)->LearningContractError {.init(message)}
}
