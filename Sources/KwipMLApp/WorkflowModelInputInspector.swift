import SwiftUI
import KwipMLCore

extension WorkflowControlNodeContext {
    var isModelDefinition:Bool {authority.node.map(WorkflowModelBoundary.isDefinition) ?? false}
    var modelInputBoundary:WorkflowNode? {try? WorkflowModelInput.boundary(for:authority.address,in:authority.owner.document,name:authority.owner.document.namedModelInputs?[authority.address.localID] == nil ? nil:selectedModelInputName)}
    var modelInputFailure:String? {
        do {_ = try WorkflowModelInput.boundary(for:authority.address,in:authority.owner.document,name:authority.owner.document.namedModelInputs?[authority.address.localID] == nil ? nil:selectedModelInputName);return nil}
        catch {return error.localizedDescription}
    }
    var hasBagInput:Bool {if case .modelPrimitive("bag_input",_,_)=modelInputBoundary?.payload{return true};return false}
    var modelInputType:String {if case .modelPrimitive(let type,_,_)=modelInputBoundary?.payload {return type};return "input"}
    var modelInputKindIsEditable:Bool {
        if case .modelPrimitive(_,let version,_)=modelInputBoundary?.payload {return version==2}
        return true
    }
    var modelInputFields:[ComponentField] {
        let config:[String:JSONValue]
        if case .modelPrimitive(_,_,let values)=modelInputBoundary?.payload {config=values} else {config=[:]}
        return authority.owner.modelRegistry.component(type:modelInputType)?.fields(configuration:config) ?? []
    }
    var modelInputConfiguration:[String:JSONValue] {
        if case .modelPrimitive(_,_,let config)=modelInputBoundary?.payload {return config}
        return Dictionary(uniqueKeysWithValues:modelInputFields.map {($0.key,$0.defaultValue)})
    }
    var modelInputConfigurationIsValid:Bool {
        let config=modelInputConfiguration
        guard Set(config.keys)==Set(modelInputFields.map(\.key)),modelInputFields.allSatisfy({field in config[field.key].map(field.accepts) ?? false}) else {return false}
        return modelInputType != "index_input" || (try? IndexInputConfiguration(values:config)) != nil
    }
    @discardableResult func resetModelInput()->Bool {
        setModelInput(configuration:Dictionary(uniqueKeysWithValues:modelInputFields.map {($0.key,$0.defaultValue)}))
    }
    var modelInputDestinations:[WorkflowEndpoint] {modelInputBoundary.map {WorkflowModelInput.destinations(of:$0,in:authority.owner.document)} ?? []}
    struct ModelInputTarget:Identifiable {
        let endpoint:WorkflowEndpoint
        let title:String
        let available:Bool
        var id:String {endpoint.address.localID+":"+endpoint.port}
    }
    var modelInputTargets:[ModelInputTarget] {
        let document=authority.owner.document,region=WorkflowModelInput.ancestors(of:authority.address,in:document)
        let boundary=modelInputBoundary
        let selected=Set(boundary.map {WorkflowModelInput.destinations(of:$0,in:document)} ?? [])
        return document.nodes.filter {node in
            node.address.domain == .model && !WorkflowModelInput.isBoundary(node) && (region.contains(node.address) || selected.contains(where:{$0.address==node.address}))
        }.flatMap {node->[ModelInputTarget] in
            guard let ports=try? WorkflowPortContracts.resolve(node,document:document,modelRegistry:authority.owner.modelRegistry,dataRegistry:authority.owner.dataRegistry) else {return []}
            return ports.inputs.filter {$0.kind == .tensor}.map {port in
                let endpoint=WorkflowEndpoint(address:node.address,port:port.id)
                let occupied=document.edges.contains {$0.to==endpoint && $0.from.address != boundary?.address}
                let name=node.name.isEmpty ? (authority.owner.modelRegistry.component(type:{if case .modelPrimitive(let type,_,_)=node.payload{return type};return ""}())?.title ?? node.address.localID) : node.name
                return .init(endpoint:endpoint,title:name+" · "+port.id,available:!occupied)
            }.filter {$0.available || selected.contains($0.endpoint)}
        }
    }
    var modelInputShared:Bool {
        guard let boundary=modelInputBoundary else {return false}
        return authority.owner.document.nodes.filter {node in
            guard node.address != authority.address,WorkflowModelBoundary.isDefinition(node) else {return false}
            return (try? WorkflowModelInput.boundary(for:node.address,in:authority.owner.document))?.address == boundary.address
        }.count>0
    }
    @discardableResult func setModelInput(type:String?=nil,configuration:[String:JSONValue]?=nil,destinations:[WorkflowEndpoint]?=nil)->Bool {
        guard isModelDefinition,!hasBagInput,modelInputFailure==nil else {return false}
        let chosen=type ?? modelInputType
        let config=configuration ?? (chosen==modelInputType ? modelInputConfiguration : Dictionary(uniqueKeysWithValues:(authority.owner.modelRegistry.component(type:chosen)?.fields ?? []).map {($0.key,$0.defaultValue)}))
        return authority.apply(.setModelInput(definition:authority.address,type:chosen,configuration:config,destinations:destinations ?? modelInputDestinations,inputName:authority.owner.document.namedModelInputs?[authority.address.localID] == nil ? nil:selectedModelInputName))
    }
}

struct WorkflowModelInputInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @State private var newInputName=""
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Model inputs").font(.headline).foregroundStyle(MLTheme.rose)
            let names=context.authority.owner.document.namedModelInputs?[context.authority.address.localID] ?? [:]
            if !names.isEmpty {
                FlatChoiceControl(title:"Input",selection:context.selectedModelInputName,options:names.keys.sorted().map {.init(id:$0,title:$0)}) {context.selectedModelInputName=$0}
            }
            HStack {
                TextField("New input name",text:$newInputName).flatField()
                Button("Add") {
                    let name=newInputName
                    if context.authority.apply(.setModelInput(definition:context.authority.address,type:"input",configuration:["features":2],destinations:[],inputName:name)) {context.selectedModelInputName=name;newInputName=""}
                }.buttonStyle(FlatButtonStyle()).disabled(newInputName.isEmpty || context.isBusy || names.count>=8)
            }
            if names.count>1 {
                Button("Remove selected input") {
                    if context.authority.apply(.removeModelInput(definition:context.authority.address,inputName:context.selectedModelInputName)) {
                        context.selectedModelInputName=(context.authority.owner.document.namedModelInputs?[context.authority.address.localID] ?? [:]).keys.sorted().first ?? "input"
                    }
                }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy || !context.modelInputDestinations.isEmpty)
            }
            if let failure=context.modelInputFailure {
                Text(failure).font(.caption).foregroundStyle(MLTheme.rose)
            } else if context.hasBagInput {
                Text("Bag packet").font(.subheadline)
                Text("Configure Bag Input on the canvas. Its index, offset and optional weight ports stay explicit.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
                FlatChoiceControl(title:"Input kind",selection:context.modelInputType,options:[.init(id:"input",title:"Floating values",isEnabled:context.modelInputKindIsEditable || context.modelInputType=="input"),.init(id:"index_input",title:"Integer indices",isEnabled:context.modelInputKindIsEditable || context.modelInputType=="index_input")]) {_=context.setModelInput(type:$0)}
                if context.modelInputType=="input",context.modelInputKindIsEditable {
                    FlatChoiceControl(title:"Floating input dimensions",selection:context.modelInputConfiguration["shape"] == nil ? "features":"tensor",options:[.init(id:"features",title:"Flat features"),.init(id:"tensor",title:"Explicit tensor shape")]) {value in
                        _=context.setModelInput(configuration:value=="tensor" ? ["shape":.array([1,2,2])]:["features":2])
                    }
                }
                ForEach(context.modelInputFields,id:\.key) {field in
                    if let value=context.modelInputConfiguration[field.key] {
                        RegistryFieldControl(field:field,value:value,origin:context.numericOrigin(key:"modelInput."+field.key),currentOrigin:{context.currentNumericOrigin(key:"modelInput."+field.key)},onDraftChange:{origin,pending in context.setNumericPending(origin:origin,key:"modelInput."+field.key,pending:pending)},componentID:context.modelInputType,configuration:context.modelInputConfiguration) {value in
                            var next=context.modelInputConfiguration;next[field.key]=value;return context.setModelInput(configuration:next)
                        }
                    }
                }
                if !context.modelInputConfigurationIsValid {
                    Button("Restore supported input defaults") {_=context.resetModelInput()}.buttonStyle(FlatButtonStyle())
                }
                Text(context.modelInputType=="input" ? (context.modelInputConfiguration["shape"] == nil ? "float32 · one feature vector per observation":"float32 · explicit per-observation shape · batch supplied at execution") : "int64 · exact category indices per observation").font(.caption).foregroundStyle(MLTheme.muted)
                Text("Entry ports").font(.subheadline.weight(.semibold))
                if context.modelInputTargets.isEmpty {
                    Text("Connect the model’s Output to this definition to choose its entry ports.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                ForEach(context.modelInputTargets) {target in
                    Toggle(target.title,isOn:Binding(get:{context.modelInputDestinations.contains(target.endpoint)},set:{enabled in
                        var next=context.modelInputDestinations
                        if enabled {next.append(target.endpoint)} else {next.removeAll {$0==target.endpoint}}
                        _=context.setModelInput(destinations:next)
                    })).toggleStyle(.checkbox).disabled(!target.available)
                    .accessibilityIdentifier("model-input.target."+target.id)
                }
                if context.modelInputShared {Text("This input is shared with another model definition. Changes apply to both.").font(.caption).foregroundStyle(MLTheme.muted)}
            }
        }.accessibilityIdentifier("model-input.configuration")
    }
}
