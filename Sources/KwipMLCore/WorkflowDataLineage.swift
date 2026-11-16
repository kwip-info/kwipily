import Foundation

/// Shared declarative partition and fitted-state lineage for training and
/// independent evaluation. Does not read or materialize any data artifacts.
enum WorkflowDataLineage {
    struct Rows:Equatable {
        var source:WorkflowNodeAddress
        var split:WorkflowNodeAddress?
        var role:String="unassigned"
        var states:[WorkflowNodeAddress]=[]
        var sequenceWindows:WorkflowNodeAddress?=nil
        var mediaWindows:WorkflowNodeAddress?=nil
    }
    enum Evidence {case rows(Rows),state(Rows)}
    static func resolve(_ region:WorkflowCompiler.DependencyRegion,dataRegistry:DataFlowOperatorRegistry)throws->[WorkflowEndpoint:Evidence] {
        let nodes=region.nodes,order=region.order,incoming=region.incoming
        func dependency(_ address:WorkflowNodeAddress,_ port:String)throws->WorkflowEndpoint {
            guard let edge=incoming[address]?.first(where:{$0.to.port==port}) else {throw workflowFailure(.requiredInput,"Missing required input.",address,port)}
            return edge.from
        }
        var evidence:[WorkflowEndpoint:Evidence]=[:]
        func rows(_ endpoint:WorkflowEndpoint,consumer:WorkflowNodeAddress)throws->Rows {
            guard case .rows(let value)=evidence[endpoint] else {throw workflowFailure(.lineage,"Input does not carry known table partition lineage.",consumer)}
            return value
        }
        let preserve:Set<String>=["kwip.data.sort","kwip.data.view","kwip.data.formula","kwip.data.columns","kwip.data.cast","kwip.data.selectRename","kwip.data.filter","kwip.data.missing","kwip.data.sampleRows",
            "kwip.data.calculate","kwip.data.regexExtract","kwip.data.regexReplace","kwip.data.outputDelimited"]
        let fit:Set<String>=["kwip.data.standardScalerFit","kwip.data.imputerFit","kwip.data.oneHotEncoderFit"]
        for address in order {
            guard let node=nodes[address],case .dataOperator(let id,let config)=node.payload else {continue}
            guard let op=dataRegistry.operatorDefinition(id:id) else {throw workflowFailure(.unsupportedOperator,"Unknown data operator.",address)}
            do {try op.validateConfiguration(config)} catch {throw workflowFailure(.configuration,error.localizedDescription,address)}
            func emit(_ port:String,_ value:Evidence){evidence[.init(address:address,port:port)]=value}
            if DataFlowSourceOperators.contains(id) {emit("table",.rows(.init(source:address)));continue}
            if id=="kwip.data.union" {
                let configuration=try UnionConfiguration(config)
                for port in configuration.inputs {
                    let input=try rows(dependency(address,port),consumer:address)
                    guard input.role=="unassigned",input.split==nil,input.states.isEmpty else {
                        throw workflowFailure(.lineage,"Union before partitioning or fitting transforms.",address,port)
                    }
                }
                emit("table",.rows(.init(source:address)));continue
            }
            if id=="kwip.data.unique" {
                let input=try rows(dependency(address,"table"),consumer:address)
                emit("unique",.rows(input));emit("duplicates",.rows(input));continue
            }
            if id=="kwip.data.reshapeTable" || id=="kwip.data.splitText" {
                let input=try rows(dependency(address,"table"),consumer:address)
                let changesPopulation=id=="kwip.data.reshapeTable" || (config["mode"] == .string("delimited") && config["outputMode"] == .string("rows"))
                if changesPopulation {
                    guard input.role=="unassigned",input.split==nil,input.states.isEmpty else {
                        throw workflowFailure(.lineage,"Reshape Table and Split Text Rows belong before Partition and fitted preprocessing.",address)
                    }
                    emit("table",.rows(.init(source:address)))
                } else {emit("table",.rows(input))}
                continue
            }
            if id=="kwip.data.aggregate" {
                let input=try rows(dependency(address,"table"),consumer:address)
                guard input.role=="unassigned",input.split==nil,input.states.isEmpty else {
                    throw workflowFailure(.lineage,"Aggregate before splitting or fitting transforms.",address)
                }
                // Group membership defines a new observation population. Preserve
                // every physical source separately in the captured ancestor set.
                emit("table",.rows(.init(source:address)));continue
            }
            if id=="kwip.data.join" {
                let left=try rows(dependency(address,"left"),consumer:address)
                let right=try rows(dependency(address,"right"),consumer:address)
                guard [left,right].allSatisfy({$0.role=="unassigned" && $0.split==nil && $0.states.isEmpty}) else {
                    throw workflowFailure(.lineage,"Join sources before splitting or fitting transforms.",address)
                }
                // A joined population has its own origin. Physical CSV bindings are
                // independently collected from all ancestors below.
                emit("table",.rows(.init(source:address)));continue
            }
            if ["kwip.data.filterRows","kwip.data.filterExpression","kwip.data.regexMatch"].contains(id) {
                // Both complementary subsets retain the same source, partition role
                // and fitted-state chain. Neither creates a new split or fit boundary.
                let input=try rows(dependency(address,"table"),consumer:address)
                emit("matching",.rows(input));emit("nonmatching",.rows(input));continue
            }
            if id==SequenceWindowsConfiguration.operatorID {
                let ports=config["mode"] == .string("partitions") ? ["training","validation","test"]:["table"]
                var primary:Rows?
                for port in ports where incoming[address]?.contains(where:{$0.to.port==port}) == true {
                    var input=try rows(dependency(address,port),consumer:address)
                    guard input.sequenceWindows==nil else {throw workflowFailure(.lineage,"Sequence windows cannot be windowed again. Connect the original token sequence.",address)}
                    if port != "table" {
                        guard input.role==port,input.split != nil else {throw workflowFailure(.lineage,"Connect the matching prepared partition.",address,port)}
                        if let primary {guard primary.source==input.source,primary.split==input.split,primary.states==input.states else {throw workflowFailure(.lineage,"Sequence partitions need the same source and fitted state.",address)}} else {primary=input}
                    }
                    input.sequenceWindows=address;emit(port,.rows(input))
                }
                continue
            }
            if id==TextEncodingConfiguration.operatorID {
                if config["vocabularySource"] == .string("supplied") {
                    let input=try rows(dependency(address,"table"),consumer:address)
                    emit("table",.rows(input));emit("state",.state(input))
                } else {
                    let training=try rows(dependency(address,"training"),consumer:address)
                    guard training.role=="training",training.split != nil else {throw workflowFailure(.lineage,"Fit text vocabulary only on Training.",address,"training")}
                    for role in ["training","validation","test"] where incoming[address]?.contains(where:{$0.to.port==role}) == true {
                        var input=try rows(dependency(address,role),consumer:address)
                        guard input.role==role,input.source==training.source,input.split==training.split,input.states==training.states else {throw workflowFailure(.lineage,"Text encoding requires matching partition roles, source and prior preparation.",address,role)}
                        input.states.append(address);emit(role,.rows(input))
                    }
                    emit("state",.state(training))
                }
                emit("vocabulary",.rows(.init(source:address)));continue
            }
            if id=="kwip.data.preprocess" {
                let configuration=try PreprocessConfiguration(values:config)
                if configuration.mode=="table" || configuration.mode=="frozen" {
                    var input=try rows(dependency(address,"table"),consumer:address)
                    if configuration.mode=="frozen" {input.states.append(address)}
                    emit("table",.rows(input));continue
                }
                let training=try rows(dependency(address,"training"),consumer:address)
                guard training.role=="training",training.split != nil else {throw workflowFailure(.lineage,"Preprocess learns only from Training.",address,"training")}
                for role in ["training","validation","test"] {
                    if incoming[address]?.contains(where:{$0.to.port==role}) != true {
                        continue
                    }
                    var input=try rows(dependency(address,role),consumer:address)
                    guard input.role==role,input.source==training.source,input.split==training.split,input.states==training.states else {throw workflowFailure(.lineage,"Preprocess needs matching source, split and prior preparation with the declared partition roles.",address,role)}
                    input.states.append(address);emit(role,.rows(input))
                }
                continue
            }
            if id==SpeechTranscriptionConfiguration.operatorID {
                let input=try rows(dependency(address,"table"),consumer:address)
                emit("transcripts",.rows(input));emit("segments",.rows(input));continue
            }
            if id==OCRConfiguration.operatorID {
                let input=try rows(dependency(address,"table"),consumer:address)
                emit("documents",.rows(input));emit("regions",.rows(input));continue
            }
            if id==PrepareMediaConfiguration.operatorID {
                var input=try rows(dependency(address,"table"),consumer:address)
                try PrepareMediaConfiguration.validate(config)
                if config["kind"] == .string("audio"),config["windowSamples"] != .integer(0) {
                    guard input.mediaWindows==nil else {throw workflowFailure(.lineage,"Prepare original recordings once; do not re-window prepared rows.",address)}
                    input.mediaWindows=address
                }
                emit("table",.rows(input));continue
            }
            if preserve.contains(id) {
                let input=try rows(dependency(address,"table"),consumer:address)
                emit("table",.rows(input));continue
            }
            if id=="kwip.data.split" {
                var input=try rows(dependency(address,"table"),consumer:address)
                guard input.mediaWindows==nil,input.sequenceWindows==nil,input.role=="unassigned",input.split==nil else {throw workflowFailure(.lineage,"Cannot split an existing partition.",address)}
                input.split=address;input.role="training";emit("training",.rows(input));input.role="validation";emit("validation",.rows(input));continue
            }
            if ["kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.partition"].contains(id) {
                let before=try rows(dependency(address,"table"),consumer:address)
                guard before.mediaWindows==nil else {throw workflowFailure(.lineage,"Partition source files before Prepare media window expansion so sibling windows stay together.",address)}
                guard before.sequenceWindows==nil else {throw workflowFailure(.lineage,"Partition documents or contiguous regions before Sequence windows.",address)}
                var input=try rows(dependency(address,"table"),consumer:address)
                guard input.role=="unassigned",input.split==nil,input.states.isEmpty else {throw workflowFailure(.lineage,"Split entities or time before partitioning or fitting transforms.",address)}
                input.split=address
                for role in ["training","validation","test"] {input.role=role;emit(role,.rows(input))}
                continue
            }
            if fit.contains(id) {
                let input=try rows(dependency(address,"training"),consumer:address)
                guard op.fitBoundary=="trainingPartitionOnly",input.role=="training",input.split != nil else {throw workflowFailure(.lineage,"Fitted state requires a training partition; validation, test or unassigned rows cannot be fitted.",address,"training")}
                emit("state",.state(input));continue
            }
            if id=="kwip.data.transform" {
                var input=try rows(dependency(address,"table"),consumer:address)
                let stateEndpoint=try dependency(address,"state")
                guard case .state(let fitted)=evidence[stateEndpoint],input.source==fitted.source,input.split==fitted.split,input.states==fitted.states,["training","validation","test"].contains(input.role),!input.states.contains(stateEndpoint.address) else {throw workflowFailure(.lineage,"Transform requires matching source, partition and prior fitted-state chain.",address)}
                input.states.append(stateEndpoint.address);emit("table",.rows(input));continue
            }
            throw workflowFailure(.unsupportedOperator,"Operator has no reviewed static lineage rule.",address)
        }
        return evidence
    }
}
