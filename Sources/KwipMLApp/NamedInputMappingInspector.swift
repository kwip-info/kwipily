import SwiftUI
import KwipMLCore

/// Train owns bindings; the model owns names, shapes and dtypes. This draft does
/// not run preparation or silently cast/resize a source column.
struct NamedInputMappingInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    var host:WorkflowWorkspaceSession?
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft:NamedInputPlan?
    @State private var expected:[String:JSONValue]=[:]
    @State private var original:NamedInputPlan?
    @State private var objective:LossConfiguration?
    @State private var originalObjective:LossConfiguration?
    @State private var failure:String?
    @State private var pendingFields=Set<String>()
    @State private var fieldGeneration=UUID()
    @State private var observedMedia:[String:[MediaArtifactReference]]=[:]
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary}
    private var schema:[DataColumnSchema] {host?.sourceProvider.recordedSchema(for:context.authority.address,in:context.authority.owner.document,inputPort:"training") ?? []}
    private var pending:Bool {draft != original || objective != originalObjective || !pendingFields.isEmpty}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Button("Use next-token mapping",action:nextTokenMapping).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).disabled(context.isBusy)
            if let plan=draft {
                if plan.sequenceObjective != nil {Text("Next-token cross entropy").fontWeight(.semibold).foregroundStyle(accent)}
                ForEach(Array(plan.inputs.enumerated()),id:\.element.name) {i,input in
                    inputSection(input,index:i,sequence:plan.sequenceObjective != nil)
                }
                Text("Objective roles").fontWeight(.semibold).foregroundStyle(accent)
                ForEach(Array(plan.roles.enumerated()),id:\.element.name) {i,role in
                    roleSection(role,index:i)
                }
                if let seq=plan.sequenceObjective {
                    columnChoice("Valid length",value:seq.lengthColumn,dtype:"int64") {draft?.sequenceObjective?.lengthColumn=$0}
                    if !seq.maskColumn.isEmpty {columnChoice("Loss mask",value:seq.maskColumn,dtype:"int64Sequence") {draft?.sequenceObjective?.maskColumn=$0}}
                }
                integer("Maximum tensor batch · MiB",key:"batchMiB",value:plan.maximumBatchBytes/1048576,range:1...256) {value in draft?.maximumBatchBytes=value*1048576}
                HStack {
                    Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending || context.isBusy || !pendingFields.isEmpty || plan.inputs.contains(where:observedMismatch))
                    Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                }
            }
            if let failure {Text(failure).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).accessibilityIdentifier("train.named-inputs")
            .onAppear(perform:reload).onChange(of:context.contextIdentity){_,_ in reload()}
            .onChange(of:draft){_,_ in context.setNumericPending(origin:context.numericOrigin(key:"namedInputPlan"),key:"namedInputPlan",pending:pending)}
            .onChange(of:objective){_,_ in context.setNumericPending(origin:context.numericOrigin(key:"namedInputPlan"),key:"namedInputPlan",pending:pending)}
            .onChange(of:host?.currentResult?.runID){_,_ in refreshObservedMedia()}
            .onChange(of:host?.currentDataPreparation?.phase){_,_ in refreshObservedMedia()}
            .onChange(of:context.authority.owner.document.revision){_,_ in refreshObservedMedia()}
            .onAppear {refreshObservedMedia()}
            .onDisappear {context.setNumericPending(origin:context.numericOrigin(key:"namedInputPlan"),key:"namedInputPlan",pending:false)}
    }
    private func roleSection(_ role:NamedInputPlan.Tensor,index:Int)->some View {
        VStack(alignment:.leading,spacing:6) {
            HStack {Text(role.name);Spacer();Text(role.dtype).foregroundStyle(accent)}
            ForEach(role.columns.indices,id:\.self) {column in roleColumn(role,index:index,column:column)}
        }.padding(7).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func roleColumn(_ role:NamedInputPlan.Tensor,index:Int,column:Int)->some View {
        let title=role.columns.count==1 ? "Column":"Column \(column+1)"
        let dtype=role.representation=="tokens" ? "int64Sequence":role.dtype
        return columnChoice(title,value:role.columns[column],dtype:dtype) {value in editRole(index) {$0.columns[column]=value}}
    }
    private func inputSection(_ input:NamedInputPlan.Tensor,index:Int,sequence:Bool)->some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {Text(input.name).fontWeight(.semibold);Spacer();Text(input.dtype).foregroundStyle(accent)}.padding(7).background(accent.opacity(0.09))
            Text("Sample shape  "+input.sampleShape.map {String($0)}.joined(separator:" × ")).foregroundStyle(MLTheme.muted)
            if !sequence || input.dtype=="int64" {representationControl(input,index:index,sequence:sequence)}
            ForEach(input.columns.indices,id:\.self) {column in
                columnChoice(input.columns.count==1 ? "Column":"Column \(column+1)",value:input.columns[column],dtype:input.representation=="prepared" ? "tensor":input.representation=="tokens" ? "int64Sequence":input.dtype) {value in editInput(index) {$0.columns[column]=value}}
            }
            if input.representation=="prepared" {
                Text(observedDescription(input)).foregroundStyle(observedMismatch(input) ? MLTheme.rose:MLTheme.muted)
                ForEach(input.sampleShape.indices,id:\.self) {axis in axisControl(input,index:index,axis:axis)}
                Text("Exact prepared dtype, axis order and sample dimensions must match. No reshape, normalization or missing mask is inferred.").foregroundStyle(MLTheme.muted)
            }
            if input.representation=="tokens" {tokenControls(input,index:index,sequence:sequence)}
        }.padding(7).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func representationControl(_ input:NamedInputPlan.Tensor,index:Int,sequence:Bool)->some View {
        let options:[FlatChoiceOption]=[.init(id:"numeric",title:"Numeric columns",isEnabled:!sequence && input.elementCount<=128),.init(id:"tokens",title:"Token sequence",isEnabled:input.dtype=="int64" && input.sampleShape.count==1),.init(id:"prepared",title:"Prepared tensor",isEnabled:!sequence)]
        return FlatChoiceControl(title:"Representation",selection:input.representation,options:options,accent:accent) {value in
            editInput(index) {row in
                row.representation=value;row.sampleAxes=nil
                if value=="prepared" {row.columns=[""];row.lengthPolicy=nil;row.paddingValue=nil;row.vocabularySize=nil;row.sampleAxes=Self.axes(rank:row.sampleShape.count)}
                else if value=="tokens" {row.columns=[row.columns.first ?? ""];row.lengthPolicy="error";row.paddingValue=0;row.vocabularySize=256}
                else {row.columns=Array(repeating:"",count:min(128,row.elementCount));row.lengthPolicy=nil;row.paddingValue=nil;row.vocabularySize=nil}
            }
        }
    }
    private func tokenControls(_ input:NamedInputPlan.Tensor,index:Int,sequence:Bool)->some View {
        let options:[FlatChoiceOption]=sequence ? [.init(id:"error",title:"Exact window length")]:[.init(id:"error",title:"Refuse"),.init(id:"pad",title:"Pad shorter"),.init(id:"truncate",title:"Truncate longer"),.init(id:"padTruncate",title:"Pad or truncate")]
        return VStack(alignment:.leading,spacing:8) {
            FlatChoiceControl(title:"Length mismatch",selection:input.lengthPolicy ?? "",options:options,accent:accent) {value in editInput(index) {$0.lengthPolicy=value}}
            integer("Vocabulary size",key:input.name+".vocabularySize",value:input.vocabularySize ?? 256,range:sequence ? 2...512:1...65536) {value in editInput(index) {$0.vocabularySize=value}}
            integer("Padding ID",key:input.name+".paddingValue",value:input.paddingValue ?? 0,range:0...max(0,(input.vocabularySize ?? 256)-1)) {value in editInput(index) {$0.paddingValue=value}}
        }
    }
    private func axisControl(_ input:NamedInputPlan.Tensor,index:Int,axis:Int)->some View {
        let title="Input "+input.name+" axis "+String(axis+1)
        let choices:[FlatChoiceOption]=["channels","samples","height","width","features","tokens","frequency","time"].map {FlatChoiceOption(id:$0,title:$0)}
        let selected=input.sampleAxes?[axis] ?? ""
        return VStack(alignment:.leading,spacing:4) {
            Text("Axis \(axis+1) · expected size \(input.sampleShape[axis])").foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:title,selection:selected,options:choices,accent:accent) {value in
                editInput(index) {row in
                    var axes=row.sampleAxes ?? Self.axes(rank:row.sampleShape.count)
                    guard axes.indices.contains(axis) else {return}
                    axes[axis]=value;row.sampleAxes=axes
                }
            }
        }
    }
    private func observedMismatch(_ input:NamedInputPlan.Tensor)->Bool {
        guard input.representation=="prepared",let name=input.columns.first else {return false}
        return (observedMedia[name] ?? []).contains {$0.dtype != input.dtype || $0.shape != input.sampleShape || $0.axes != input.sampleAxes}
    }
    private func observedDescription(_ input:NamedInputPlan.Tensor)->String {
        guard let name=input.columns.first,let values=observedMedia[name],let first=values.first else {return "No current captured sample · run preparation to inspect actual geometry."}
        let distinct=Set(values.map {$0.dtype+":"+$0.shape.map(String.init).joined(separator:",")+":"+$0.axes.joined(separator:",")})
        if distinct.count>1 {return "Captured sample has multiple geometries. All observations must match the model contract."}
        return "Captured sample · "+first.dtype+" · "+first.shape.map(String.init).joined(separator:" × ")+" · "+first.axes.joined(separator:" × ")+(observedMismatch(input) ? " · mismatch":" · matches expected")
    }
    private func refreshObservedMedia() {
        guard let host else {observedMedia=[:];return}
        let edges=host.owner.document.edges.filter {$0.to.address==context.authority.address && $0.to.port=="training"}
        let snapshots=host.snapshots(for:Set(edges.map {$0.from.address}))
        var result:[String:[MediaArtifactReference]]=[:]
        for edge in edges {
            guard let capture=snapshots[edge.from.address],capture.matchesCurrentConstruction,capture.matchesCurrentBindings,
                  case .object(let value)?=capture.ports[edge.from.port],case .object(let preview)?=value["preview"],
                  case .array(let columns)?=preview["columns"],case .array(let rows)?=preview["rows"] else {continue}
            for (index,column) in columns.enumerated() {
                guard case .object(let fields)=column,fields["dtype"] == .string("tensor"),let name=fields["name"]?.stringValue else {continue}
                for case .array(let row) in rows.prefix(8) where row.indices.contains(index) {
                    if let bytes=try? JSONEncoder().encode(row[index]),let ref=try? MediaArtifactReference.decode(bytes) {result[name,default:[]].append(ref)}
                }
            }
        }
        observedMedia=result
    }
    private func editInput(_ index:Int,_ edit:(inout NamedInputPlan.Tensor)->Void) {guard var plan=draft,plan.inputs.indices.contains(index) else {return};edit(&plan.inputs[index])
        if plan.sequenceObjective?.inputName==plan.inputs[index].name {
            let x=plan.inputs[index]
            plan.roles[0].lengthPolicy=x.lengthPolicy;plan.roles[0].paddingValue=x.paddingValue;plan.roles[0].vocabularySize=x.vocabularySize
        }
        if plan.sequenceObjective==nil {plan.schemaVersion=plan.inputs.contains(where:{$0.representation=="prepared"}) ? 3:1}
        draft=plan}
    private static func axes(rank:Int)->[String] {if rank==1 {return ["features"]};if rank==2 {return ["channels","samples"]};if rank==3 {return ["channels","height","width"]};return Array(["channels","height","width","samples","features","tokens","frequency"].prefix(rank))}
    private func editRole(_ index:Int,_ edit:(inout NamedInputPlan.Tensor)->Void) {guard var plan=draft,plan.roles.indices.contains(index) else {return};edit(&plan.roles[index]);draft=plan}
    @ViewBuilder private func columnChoice(_ title:String,value:String,dtype:String,change:@escaping(String)->Void)->some View {
        let choices=schema.filter {$0.dtype==dtype || dtype=="float32" && ["int64","float64"].contains($0.dtype)}.map(\.name)
        if schema.isEmpty {VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{value},set:change)).flatField()}}
        else {VStack(alignment:.leading,spacing:4) {Text(title).foregroundStyle(MLTheme.muted);FlatChoiceControl(title:title,selection:value,options:([value].filter {!$0.isEmpty && !choices.contains($0)}+choices).map {.init(id:$0,title:$0)},accent:accent,onSelect:change)}}
    }
    private func integer(_ title:String,key:String,value:Int,range:ClosedRange<Int>,change:@escaping(Int)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).foregroundStyle(MLTheme.muted)
            BoundedNumberField(title:title,value:.wholeDisplay(Double(value)),domain:.integer(minimum:Double(range.lowerBound),maximum:Double(range.upperBound)),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{origin,pending in
                if pending {pendingFields.insert(key)} else {pendingFields.remove(key)}
                context.setNumericPending(origin:origin,key:key,pending:pending)
            }) {candidate in guard case .integer(let n)=candidate else {return false};change(Int(n));return true}.id(fieldGeneration)
        }
    }
    private func reload() {
        fieldGeneration=UUID();pendingFields=[];expected=context.configuration;failure=nil
        do {
            objective=try WorkflowTrainRecipes(configuration:context.configuration,schemaVersion:context.authority.owner.document.schemaVersion).objective
            originalObjective=objective
            if case .object(let values)=context.mappingConfiguration["namedInputPlan"] {let plan=try NamedInputPlan(values:values);draft=plan;original=plan;return}
            let owner=context.authority.owner,document=owner.document
            guard let definition=document.edges.first(where:{$0.to.address==context.authority.address && $0.to.port=="model"})?.from.address else {throw DataFlowContractError("Connect a Model definition before mapping tensors.")}
            let graph=try WorkflowCompiler.modelSnapshot(document,definition:definition,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,context:objective == nil ? .preparedWorkflow:.explicitObjective).graph
            let validation=GraphValidator.validate(graph,registry:owner.modelRegistry,context:objective == nil ? .preparedWorkflow:.explicitObjective)
            guard validation.isValid else {throw DataFlowContractError("Repair the connected model before mapping tensors.")}
            let sources=graph.nodes.filter {["input","index_input"].contains($0.type)}
            let inputs=try sources.map {node->NamedInputPlan.Tensor in
                guard let actual=validation.tensorContracts[node.id] else {throw DataFlowContractError("The connected input has no verified tensor contract.")}
                let name=graph.namedInputs?.first(where:{$0.value==node.id})?.key ?? "input"
                let tokenSource=actual.dtype == .int64 && actual.layout.sample.dimensions.count==1 && schema.contains {$0.dtype=="int64Sequence"}
                let prepared=schema.contains {$0.dtype=="tensor"} || (actual.layout.sample.elementCount>128 && !tokenSource)
                if prepared {return .init(name:name,nodeID:node.id,representation:"prepared",dtype:actual.dtype.rawValue,sampleShape:actual.layout.sample.dimensions,columns:[schema.first(where:{$0.dtype=="tensor"})?.name ?? ""],sampleAxes:Self.axes(rank:actual.layout.sample.dimensions.count))}
                let tokens=actual.dtype == .int64 && schema.contains {$0.dtype=="int64Sequence"}
                let columns=tokens ? [schema.first(where:{$0.dtype=="int64Sequence"})!.name]:Array(schema.filter {actual.dtype == .float32 ? ["float64","int64"].contains($0.dtype):$0.dtype=="int64"}.prefix(actual.layout.sample.elementCount).map(\.name))
                let required=tokens ? 1:actual.layout.sample.elementCount
                return .init(name:name,nodeID:node.id,representation:tokens ? "tokens":"numeric",dtype:actual.dtype.rawValue,sampleShape:actual.layout.sample.dimensions,columns:columns+Array(repeating:"",count:max(0,required-columns.count)),lengthPolicy:tokens ? "error":nil,paddingValue:tokens ? 0:nil,vocabularySize:tokens ? 256:nil)
            }
            let roles:[NamedInputPlan.Tensor]
            if let model=try? LossTargetMappingInspectorContext(context:context),objective != nil {roles=model.draft.mapping.roles.map {.init(name:$0.role,dtype:$0.dtype.rawValue,sampleShape:$0.sampleShape,columns:$0.columns)}}
            else {roles=[.init(name:"target",dtype:"int64",sampleShape:[],columns:[context.mappingConfiguration["targetColumn"]?.stringValue ?? ""])]}
            var plan=NamedInputPlan(inputs:inputs,roles:roles);if inputs.contains(where:{$0.representation=="prepared"}) {plan.schemaVersion=3};draft=plan;original=nil
        }catch {draft=nil;failure=error.localizedDescription}
    }
    private func nextTokenMapping() {
        do {
            let owner=context.authority.owner,d=owner.document
            guard let edge=d.edges.first(where:{$0.to.address==context.authority.address && $0.to.port=="training"}),edge.from.port=="training",let node=d.nodes.first(where:{$0.address==edge.from.address}),case .dataOperator(let id,let c)=node.payload,id==SequenceWindowsConfiguration.operatorID,c["mode"] == .string("partitions") else {throw DataFlowContractError("Connect Training and Validation directly from Sequence windows in Partitions mode.")}
            guard d.edges.contains(where:{$0.to.address==context.authority.address && $0.to.port=="validation" && $0.from.address==node.address && $0.from.port=="validation"}),let definition=d.edges.first(where:{$0.to.address==context.authority.address && $0.to.port=="model"})?.from.address else {throw DataFlowContractError("Connect the matching Validation and Model inputs.")}
            let graph=try WorkflowCompiler.modelSnapshot(d,definition:definition,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,context:.explicitObjective).graph
            let validation=GraphValidator.validate(graph,registry:owner.modelRegistry,context:.explicitObjective)
            let boundaries=graph.nodes.filter {["input","index_input"].contains($0.type)}
            guard validation.isValid,boundaries.count==1,let input=boundaries.first,let actual=validation.tensorContracts[input.id],actual.dtype == .int64,actual.layout.sample.dimensions==[c["contextLength"]!.integerValue!],let output=graph.nodes.first(where:{$0.type=="output"}),let produced=validation.tensorContracts[output.id],produced.dtype == .float32,produced.layout.sample.dimensions.count==2,produced.layout.sample.dimensions[0]==actual.layout.sample.dimensions[0],(2...512).contains(produced.layout.sample.dimensions[1]) else {throw DataFlowContractError("This preset requires one integer token input [context] and float logits [context, vocabulary] with 2–512 classes. Configure the model first.")}
            let name=graph.namedInputs?.first(where:{$0.value==input.id})?.key ?? "input",vocabulary=produced.layout.sample.dimensions[1],padding=c["paddingID"]!.integerValue!
            var p=NamedInputPlan(inputs:[.init(name:name,nodeID:input.id,representation:"tokens",dtype:"int64",sampleShape:actual.layout.sample.dimensions,columns:[c["inputColumn"]!.stringValue!],lengthPolicy:"error",paddingValue:padding,vocabularySize:vocabulary)],roles:[.init(name:"target",representation:"tokens",dtype:"int64",sampleShape:actual.layout.sample.dimensions,columns:[c["targetColumn"]!.stringValue!],lengthPolicy:"error",paddingValue:padding,vocabularySize:vocabulary)])
            p.schemaVersion=2;p.sequenceObjective = .init(inputName:name,lengthColumn:c["lengthColumn"]!.stringValue!,maskColumn:c["incomplete"] == .string("pad") ? c["maskColumn"]!.stringValue!:"")
            try p.validate();fieldGeneration=UUID();pendingFields=[];draft=p;objective=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]);failure=nil
        } catch {failure=error.localizedDescription}
    }
    private func apply() {
        do {
            guard let plan=draft,context.configuration==expected else {throw DataFlowContractError("Saved configuration changed. Revert to load it; your draft is preserved.")}
            try plan.validate()
            let columns=plan.inputs.flatMap(\.columns).reduce(into:[String]()) {if !$0.contains($1) {$0.append($1)}}
            var mapping:[String:JSONValue]=["namedInputPlan":.object(plan.values)]
            if let objective {
                let roles=try plan.roles.map {role->LossTargetRoleMapping in guard let dtype=ContractDType(rawValue:role.dtype) else {throw DataFlowContractError("Unsupported objective dtype.")};return .init(role:role.name,columns:role.columns,sampleShape:role.sampleShape,dtype:dtype)}
                var target=LossTargetMapping(componentID:objective.componentID,inputColumns:columns,roles:roles)
                if plan.sequenceObjective != nil {target.schemaVersion=2}
                try objective.validate(targetMapping:target)
                mapping["lossTargetMapping"]=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(target))
            }else {
                guard plan.roles.count==1,let target=plan.roles.first,target.name=="target",target.dtype=="int64",target.sampleShape.isEmpty,target.columns.count==1 else {throw DataFlowContractError("Choose a scalar integer target or configure an explicit objective.")}
                mapping["featureColumns"] = .array(columns.map(JSONValue.string));mapping["targetColumn"] = .string(target.columns[0])
            }
            let saved = plan.sequenceObjective != nil && objective != nil ? context.configureSequenceMapping(mapping,objective:objective!,expected:expected):context.configureMapping(mapping,expected:expected)
            if saved {original=plan;originalObjective=objective;expected=context.configuration;failure=nil;context.setNumericPending(origin:context.numericOrigin(key:"namedInputPlan"),key:"namedInputPlan",pending:false)} else {failure=context.statusMessage}
        }catch {failure=error.localizedDescription}
    }
}
