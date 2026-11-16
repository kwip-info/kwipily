import Foundation
import Combine
import KwipMLCore

struct InitializationOwnerChoice:Identifiable,Equatable {
    let node:String
    let parameter:String
    let name:String
    let shape:[Int]
    var id:String {"\(node.utf8.count):\(node)\(parameter.utf8.count):\(parameter)"}
    var title:String {name+" · "+TrainLayerPresentation.parameterTitle(parameter)+" · "+shape.map(String.init).joined(separator:" × ")}
}
/// Shared native-owner resolution for placement and retained editing; no invented owners.
enum InitializationOwnerResolver {
    static func choices(document:WorkflowDocument,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,recipeAddress:WorkflowNodeAddress?=nil,selectedTrain:WorkflowNodeAddress?=nil)->[InitializationOwnerChoice] {
        let connectedTrains=recipeAddress.map {address in document.edges.filter {$0.from.address == address && $0.to.port == "initialization"}.map { $0.to.address }} ?? []
        let trainNodes=document.nodes.filter {if case .train=$0.payload {return true};return false}.map(\.address)
        let train:WorkflowNodeAddress? = connectedTrains.count == 1 ? connectedTrains[0] : (selectedTrain.flatMap {trainNodes.contains($0) ? $0:nil} ?? (trainNodes.count == 1 ? trainNodes[0]:nil))
        guard connectedTrains.count <= 1 else {return []}
        let definitions=document.nodes.filter {WorkflowModelBoundary.isVisibleDefinition($0,in:document)}.map(\.address)
        let definition:WorkflowNodeAddress? = train != nil ? document.edges.first {$0.to.address==train! && $0.to.port=="model"}?.from.address : definitions.count==1 ? definitions[0]:nil
        let explicit=train != nil && document.edges.contains {$0.to.address==train! && $0.to.port=="objective"}
        let context:GraphValidationContext=explicit ? .explicitObjective:.preparedWorkflow
        return (try? definition.map { definition in
            let model=try WorkflowCompiler.modelSnapshot(document,definition:definition,modelRegistry:modelRegistry,dataRegistry:dataRegistry,context:context)
            let names=TrainLayerPresentation.names(for:model.graph,origins:model.origins,document:document,registry:modelRegistry)
            return try WorkflowLifecycleVerifier.optimizerParameterOwnership(model.graph,context:context,capturedRegistry:modelRegistry,initializationOwnership:true).map {raw in
                guard case .object(let row)=raw,let node=row["node"]?.stringValue,let parameter=row["parameter"]?.stringValue,row["name"]?.stringValue != nil,case .array(let shape)=row["shape"] else {throw WorkflowPhaseError("Weights and biases are unavailable.")}
                return InitializationOwnerChoice(node:node,parameter:parameter,name:names[node] ?? "Layer",shape:try shape.map {guard let n=$0.integerValue,let extent=Int(exactly:n) else {throw WorkflowPhaseError("Weight and bias dimensions are unavailable.")};return extent})
            }
        }) ?? []
    }
}

/// One retained rules editor; workflow and lessons supply their own existing command authority.
@MainActor final class InitializationRecipeInspectorContext:ObservableObject {
    let owners:[InitializationOwnerChoice]
    let origin:String
    @Published private(set) var draftConfiguration:[String:JSONValue]
    @Published private(set) var seedText:String
    @Published private(set) var rejection:String?
    private(set) var expectedConfiguration:[String:JSONValue]
    private var baseline:[String:JSONValue]
    private let current:()->Bool,busy:()->Bool
    private let commit:([String:JSONValue],[String:JSONValue])->Bool
    private let pending:(Bool)->Void
    private var dormant:[Int:[String:[String:JSONValue]]]=[:]
    private var numericPending=Set<String>()
    @Published private(set) var rawNumericDrafts:[String:String]=[:]
    private func numericValue(_ key:String,in configuration:[String:JSONValue]?=nil)->Double? {
        let pieces=key.split(separator:".")
        guard pieces.count==3,pieces[0]=="rule",let index=Int(pieces[1]),["mean","std","gain","a"].contains(String(pieces[2])),case .array(let rows)=(configuration ?? draftConfiguration)["rules"],rows.indices.contains(index),case .object(let rule)=rows[index],case .object(let settings)=rule["settings"] else {return nil}
        return settings[String(pieces[2])]?.doubleValue
    }
    private var hasRawPending:Bool {rawNumericDrafts.contains {key,text in guard let value=numericValue(key) else {return true};return text != NumericValue.decimal(value).text}}
    @discardableResult func restorePending(configuration:[String:JSONValue],seedText:String,rawNumericDrafts:[String:String]=[:])->Bool {
        guard !isBusy,Set(configuration.keys)==["seed","rules"],case .integer(let seed)=configuration["seed"],seed>=0,case .array(let rows)=configuration["rules"],rows.count<=128,seedText.utf8.count<=64,rawNumericDrafts.count<=384,let encoded=try? JSONEncoder().encode(configuration),encoded.count<=32768 else {return false}
        for row in rows {guard (try? InitializationConfiguration(values:["seed":0,"rules":.array([row])])) != nil else {return false}}
        guard rawNumericDrafts.allSatisfy({key,text in numericValue(key,in:configuration) != nil && text.utf8.count<=256}) else {return false}
        self.draftConfiguration=configuration;self.seedText=seedText;self.rawNumericDrafts=rawNumericDrafts
        numericPending.removeAll();dormant.removeAll();resetGeneration+=1;rejection=nil;report();return true
    }
    @discardableResult func setRawNumericDraft(_ key:String,text:String,origin:String)->Bool {
        guard !isBusy,currentNumericOrigin(key)==origin,text.utf8.count<=256,let value=numericValue(key) else {return false}
        if text==NumericValue.decimal(value).text {rawNumericDrafts.removeValue(forKey:key)}else{rawNumericDrafts[key]=text}
        report();return true
    }
    private(set) var resetGeneration=0
    init(configuration:[String:JSONValue],owners:[InitializationOwnerChoice],origin:String,
         isCurrent:@escaping()->Bool,isBusy:@escaping()->Bool,
         apply:@escaping([String:JSONValue],[String:JSONValue])->Bool,pending:@escaping(Bool)->Void) {
        self.owners=owners;self.origin=origin;current=isCurrent;busy=isBusy;commit=apply;self.pending=pending
        baseline=configuration;expectedConfiguration=configuration;draftConfiguration=configuration
        seedText=configuration["seed"]?.integerValue.map(String.init) ?? "42"
    }
    var isBusy:Bool {!current() || busy()}
    var hasChanges:Bool {draftConfiguration != baseline || seedText != baseline["seed"]?.integerValue.map(String.init)}
    var rules:[[String:JSONValue]] {guard case .array(let rows)=draftConfiguration["rules"] else {return []};return rows.compactMap {if case .object(let row)=$0 {return row};return nil}}
    private func canonical()throws->InitializationConfiguration {
        guard !seedText.isEmpty,seedText.utf8.allSatisfy({(48...57).contains($0)}),let seed=Int64(seedText),seed>=0 else {throw GraphConfigurationError("Seed must be a whole number from 0 through 9223372036854775807.")}
        var config=draftConfiguration;config["seed"] = .integer(seed)
        let value=try InitializationConfiguration(values:config)
        for rule in value.rules {
            guard let owner=owners.first(where:{$0.node==rule.owner.node && $0.parameter==rule.owner.parameter}) else {throw GraphConfigurationError("Choose current weights or biases from the model connected to Train. Connect the model before configuring starting settings.")}
            if ["xavier_uniform","kaiming_uniform"].contains(rule.algorithm),owner.shape.count<2 {throw GraphConfigurationError("Xavier and Kaiming need a weight array with at least two directions. Use Normal or Zeros for a bias array.")}
        }
        return value
    }
    func validatedConfiguration() throws -> [String: JSONValue] {
        guard !hasUncommittedNumbers else { throw GraphConfigurationError("Apply or revert pending numeric values first.") }
        return try canonical().values
    }
    var validationMessage:String? {do {_=try canonical();return nil}catch{return error.localizedDescription}}
    var hasUncommittedNumbers:Bool {!numericPending.isEmpty || hasRawPending}
    var canApply:Bool {!isBusy && hasChanges && !hasUncommittedNumbers && validationMessage==nil}
    private func report() {pending(hasChanges || hasUncommittedNumbers)}
    @discardableResult func setSeedText(_ text:String)->Bool {
        guard !isBusy,text.utf8.count<=64 else {return false};seedText=text;rejection=nil;report();return true
    }
    private func replace(_ row:[String:JSONValue],at index:Int)->Bool {
        guard !isBusy,rules.indices.contains(index) else {return false}
        var updated=rules;updated[index]=row;draftConfiguration["rules"] = .array(updated.map { .object($0) });rejection=nil;report();return true
    }
    @discardableResult func setOwner(_ id:String,at index:Int)->Bool {
        guard !isBusy,rules.indices.contains(index),let owner=owners.first(where:{$0.id==id}) else {return false}
        var row=rules[index];row["owner"] = .object(["node":.string(owner.node),"parameter":.string(owner.parameter)])
        return replace(row,at:index)
    }
    @discardableResult func setAlgorithm(_ algorithm:String,at index:Int)->Bool {
        guard !isBusy,!hasUncommittedNumbers,rules.indices.contains(index),let previous=rules[index]["algorithm"]?.stringValue,case .object(let settings)=rules[index]["settings"] else {return false}
        let defaults:[String:[String:JSONValue]]=["normal":["mean":0,"std":0.02],"zeros":[:],"xavier_uniform":["gain":1],"kaiming_uniform":["a":0,"mode":.string("fan_in"),"nonlinearity":.string("leaky_relu")]]
        guard let initial=defaults[algorithm] else {return false}
        dormant[index,default:[:]][previous]=settings
        var row=rules[index];row["algorithm"] = .string(algorithm);row["settings"] = .object(dormant[index]?[algorithm] ?? initial)
        return replace(row,at:index)
    }
    @discardableResult func setSetting(_ key:String,value:JSONValue,at index:Int)->Bool {
        guard !isBusy,rules.indices.contains(index),case .object(var settings)=rules[index]["settings"],settings[key] != nil else {return false}
        settings[key]=value;var row=rules[index];row["settings"] = .object(settings)
        guard (try? InitializationConfiguration(values:["seed":0,"rules":.array([.object(row)])])) != nil else {return false}
        return replace(row,at:index)
    }
    @discardableResult func addOwner()->Bool {
        guard !isBusy,!hasUncommittedNumbers,rules.count<128,let choice=owners.first(where:{candidate in !rules.contains {row in guard case .object(let owner)=row["owner"] else {return false};return owner["node"] == .string(candidate.node) && owner["parameter"] == .string(candidate.parameter)}}) else {return false}
        var updated=rules;updated.append(["owner":.object(["node":.string(choice.node),"parameter":.string(choice.parameter)]),"algorithm":.string("normal"),"settings":.object(["mean":0,"std":0.02])])
        draftConfiguration["rules"] = .array(updated.map { .object($0) });report();return true
    }
    @discardableResult func removeRule(at index:Int)->Bool {
        guard !isBusy,!hasUncommittedNumbers,rules.indices.contains(index) else {return false};var updated=rules;updated.remove(at:index)
        draftConfiguration["rules"] = .array(updated.map { .object($0) });dormant.removeAll();resetGeneration+=1;report();return true
    }
    func numericOrigin(_ key:String)->String {origin+":"+String(resetGeneration)+":"+key}
    func currentNumericOrigin(_ key:String)->String? {isBusy ? nil:numericOrigin(key)}
    func setNumericPending(_ key:String,origin:String,pending:Bool) {
        guard current(),origin==numericOrigin(key) else {return};if pending {numericPending.insert(key)}else{numericPending.remove(key)};report()
    }
    @discardableResult func apply()->Bool {
        guard !isBusy,!hasUncommittedNumbers else {return false}
        do {
            let config=try canonical().values
            guard hasChanges else {return true}
            guard commit(expectedConfiguration,config) else {rejection="The source changed. Reopen this editor before applying the draft.";return false}
            expectedConfiguration=config;baseline=config;draftConfiguration=config;seedText=config["seed"]!.integerValue.map(String.init)!
            dormant.removeAll();rawNumericDrafts.removeAll();resetGeneration+=1;rejection=nil;report();return true
        }catch{rejection=error.localizedDescription;return false}
    }
    @discardableResult func discard()->Bool {
        guard !isBusy else {return false};draftConfiguration=baseline;seedText=baseline["seed"]?.integerValue.map(String.init) ?? "42"
        numericPending.removeAll();rawNumericDrafts.removeAll();dormant.removeAll();resetGeneration+=1;rejection=nil;report();return true
    }
    func releasePendingEdits() {numericPending.removeAll();pending(false)}
}
