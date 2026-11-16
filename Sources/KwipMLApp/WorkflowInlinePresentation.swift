import Foundation
import KwipMLCore

/// Local canvas presentation, excluded from portable source, scientific identity and worker captures.
struct WorkflowInlinePresentation:Codable,Equatable {
    struct Point:Codable,Equatable {var x:Double;var y:Double}
    struct Size:Codable,Equatable {var width:Double;var height:Double}
    var sizes:[String:Size]?=nil
    var collapsed:Set<String>=[]
    var positions:[String:Point]=[:]
    static func load(projectID:String)->Self {
        guard let bytes=UserDefaults.standard.data(forKey:"workflow.inlineContainers."+projectID),bytes.count<=512_000,
              let value=try? JSONDecoder().decode(Self.self,from:bytes),value.collapsed.count<=256,value.positions.count<=4096,
              (value.sizes?.count ?? 0)<=256,(value.sizes?.values.allSatisfy {$0.width.isFinite && $0.height.isFinite && (1...100_000).contains($0.width) && (1...100_000).contains($0.height)} ?? true),
              value.positions.values.allSatisfy({$0.x.isFinite && $0.y.isFinite && abs($0.x)<1_000_000 && abs($0.y)<1_000_000}) else {return .init()}
        return value
    }
    func save(projectID:String) {
        guard collapsed.count<=256,positions.count<=4096,let bytes=try? JSONEncoder().encode(self),bytes.count<=512_000 else {return}
        UserDefaults.standard.set(bytes,forKey:"workflow.inlineContainers."+projectID)
    }
}

extension WorkflowCanvasAdapter {
    func makeInlineProjection(_ base:WorkflowCanvasProjection,document:WorkflowDocument)throws->WorkflowCanvasProjection {
        mixtureInlinePorts=[:]
        let roots=document.nodes.filter {node in
            guard let definition=try? WorkflowInlineContainer.definition(in:document,root:node.address) else {return false}
            return definition.body.instances.isEmpty && (!definition.body.nodes.isEmpty || WorkflowMixtureContainer.contains(document,root:node.address))
        }
        let expanded=Set(roots.filter {!inlineState.collapsed.contains($0.address.localID)}.map(\.address))
        let trial=try base.addingInlineMembers(document,expandedRoots:expanded,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        let geometry=CanvasGeometrySnapshot(registry:trial.registry,presentation:canvasViewState.presentation,graph:trial.graph,validation:.init(issues:[]))
        let camera=geometry.projection
        var positions:[WorkflowNodeAddress:CGPoint]=[:],frames:[CanvasInlineContainer]=[]
        for root in roots {
            let definition=try WorkflowInlineContainer.definition(in:document,root:root.address)
            guard let id=base.displayIDByAddress[root.address] else {continue}
            let origin=camera.project(CGPoint(x:root.x,y:root.y))
            let mixtureChains=(try? WorkflowMixtureContainer.chains(in:document,root:root.address))?.filter{$0.key != "router"}
            if let mixtureChains {
                for chain in mixtureChains {
                    let checked=GraphValidator.validate(chain.graph,registry:modelRegistry,context:.explicitObjective)
                    for node in chain.graph.nodes {
                        let address=WorkflowInlineContainer.address(root:root.address,childID:WorkflowMixtureContainer.qualified(chain.id,node.id))
                        guard let metadata=modelRegistry.component(type:node.type) else{continue}
                        for (direction,names) in [(PortDirection.input,metadata.configuredInputs(node.config)),(.output,metadata.configuredOutputs(node.config))] {
                            for port in names {
                                mixtureInlinePorts[inlinePortKey(address,port,direction)]=checked.tensorPortDescriptor(nodeID:node.id,port:port,direction:direction,graph:chain.graph,registry:modelRegistry)
                            }
                        }
                    }
                }
            }
            var depth:[String:Int]=[:]
            func level(_ child:String,_ visiting:Set<String>=[])->Int {
                if let value=depth[child] {return value}
                guard !visiting.contains(child) else {return 0}
                let incoming=definition.body.edges.filter {$0.target==child}.map(\.source)
                let value=incoming.map {level($0,visiting.union([child]))+1}.max() ?? 0
                depth[child]=min(value,128);return min(value,128)
            }
            var rows:[Int:Int]=[:],childIDs:[String]=[]
            var maxX:CGFloat=500,maxY:CGFloat=240
            for child in definition.body.nodes.sorted(by:{$0.y == $1.y ? $0.id<$1.id:$0.y<$1.y}) {
                let address=WorkflowInlineContainer.address(root:root.address,childID:child.id)
                let display=try WorkflowCanvasIDs.nodeID(address);childIDs.append(display)
                let column=level(child.id)
                let chainOrder=mixtureChains?.map(\.id)
                let row:Int
                if let chainOrder,let chain=try? WorkflowMixtureContainer.split(child.id),let i=chainOrder.firstIndex(of:chain.chain){row=i}
                else{row=rows[column,default:0];rows[column]=row+1}
                let defaultOffset=CGPoint(x:200+CGFloat(column)*325,y:95+CGFloat(row)*225)
                let offset:CGPoint
                if let stored=inlineState.positions[address.localID] {offset=camera.project(CGPoint(x:stored.x,y:stored.y))}
                else {offset=defaultOffset}
                let world=camera.inverse(CGPoint(x:origin.x+offset.x,y:origin.y+offset.y))
                positions[address]=world
                let size=geometry.size(nodeID:display)
                maxX=max(maxX,offset.x+size.width+200);maxY=max(maxY,offset.y+size.height+65)
            }
            maxY=max(maxY,CanvasNodeGeometry.headerHeight+100+CGFloat(max(definition.inputs.count,definition.outputs.count))*58)
            if WorkflowMixtureContainer.contains(document,root:root.address){maxX=max(maxX,700)}
            if let mixtureChains {maxY=max(maxY,CGFloat(mixtureChains.count)*225+100)}
            let previous=inlineState.sizes?[root.address.localID]
            let size=CGSize(width:max(maxX,previous?.width ?? 0),height:max(maxY,previous?.height ?? 0))
            let retained=WorkflowInlinePresentation.Size(width:size.width,height:size.height)
            if previous != retained {inlineState.sizes=(inlineState.sizes ?? [:]).merging([root.address.localID:retained]) {_,new in new};inlineState.save(projectID:owner.projectID)}
            var input:[String:CGPoint]=[:],output:[String:CGPoint]=[:],links:[CanvasInlineContainer.BoundaryLink]=[]
            for (index,port) in definition.inputs.enumerated() where mixtureChains == nil {
                input[port.id]=CGPoint(x:0,y:CanvasNodeGeometry.headerHeight+70+CGFloat(index)*58)
                for target in port.targets {
                    links.append(.init(containerPort:port.id,childNodeID:try WorkflowCanvasIDs.nodeID(WorkflowInlineContainer.address(root:root.address,childID:target.componentID)),childPort:target.portID,direction:.input))
                }
            }
            for (index,port) in definition.outputs.enumerated() {
                output[port.id]=CGPoint(x:size.width,y:CanvasNodeGeometry.headerHeight+70+CGFloat(index)*58)
                links.append(.init(containerPort:port.id,childNodeID:try WorkflowCanvasIDs.nodeID(WorkflowInlineContainer.address(root:root.address,childID:port.source.componentID)),childPort:port.source.portID,direction:.output))
            }
            if WorkflowMixtureContainer.contains(document,root:root.address){
                for chain in mixtureChains ?? [] {
                    let entries=Set(chain.graph.nodes.filter{$0.type=="input"}.map(\.id))
                    let exits=Set(chain.graph.nodes.filter{$0.type=="output"}.map(\.id))
                    for edge in chain.graph.edges where entries.contains(edge.source) && !exits.contains(edge.target) {
                        links.append(.init(containerPort:"in",childNodeID:try WorkflowCanvasIDs.nodeID(WorkflowInlineContainer.address(root:root.address,childID:WorkflowMixtureContainer.qualified(chain.id,edge.target))),childPort:edge.targetPort,direction:.input,routed:true,containerNodeID:id))
                    }
                    for edge in chain.graph.edges where exits.contains(edge.target) && !entries.contains(edge.source) {
                        links.append(.init(containerPort:"out",childNodeID:try WorkflowCanvasIDs.nodeID(WorkflowInlineContainer.address(root:root.address,childID:WorkflowMixtureContainer.qualified(chain.id,edge.source))),childPort:edge.sourcePort,direction:.output,routed:true,containerNodeID:id))
                    }
                }
                input["in"]=CGPoint(x:0,y:CanvasNodeGeometry.headerHeight+70)
                input["mask"]=CGPoint(x:0,y:CanvasNodeGeometry.headerHeight+128)
                output["out"]=CGPoint(x:size.width,y:CanvasNodeGeometry.headerHeight+70)
            }
            var frame=CanvasInlineContainer(id:id,isExpanded:expanded.contains(root.address),size:size,childNodeIDs:childIDs,boundaryLinks:links,inputOffsets:input,outputOffsets:output)
            frame.lanes=(mixtureChains ?? []).enumerated().map{index,chain in .init(id:chain.id,title:chain.name+(chain.graph.nodes.allSatisfy{["input","output"].contains($0.type)} ? " · Identity (input passes through)":""),y:65+CGFloat(index)*225)}
            frames.append(frame)
        }
        inlineContainers=frames
        return try base.addingInlineMembers(document,expandedRoots:expanded,positions:positions,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
    }
    func fitInlineContainer(_ id:String) {
        guard !isBusy,!owner.hasPendingInspectorEdits,let address=projection.addressByDisplayID[id],inlineContainers.contains(where:{$0.id==id && $0.isExpanded}) else {return}
        inlineState.sizes?[address.localID]=nil
        refreshInlinePresentation();inlineState.save(projectID:owner.projectID)
    }
    func toggleInlineContainer(_ id:String) {
        guard !isBusy,!owner.hasPendingInspectorEdits,let address=projection.addressByDisplayID[id],inlineContainers.contains(where:{$0.id==id}) else {return}
        if inlineState.collapsed.contains(address.localID) {inlineState.collapsed.remove(address.localID)}
        else {
            inlineState.collapsed.insert(address.localID)
            if owner.selection.nodeAddresses.contains(where:{(try? WorkflowInlineContainer.parse(in:owner.document,address:$0))?.root==address}) || owner.selection.edgeIDs.contains(where:{WorkflowInlineContainerProjection.internalEdge(in:owner.document,displayID:$0)?.root==address}) {
                _=owner.select(.init(nodeAddresses:[address]),lease:owner.lease())
            }
        }
        inlineState.save(projectID:owner.projectID);pendingSourceID=nil;refreshInlinePresentation()
    }
    func canvasMovingNodeIDs(for ids:Set<String>)->Set<String> {
        var result=ids
        for container in inlineContainers where container.isExpanded && ids.contains(container.id) {result.formUnion(container.childNodeIDs)}
        return result
    }
    func inlineDropRoot(at point:CGPoint,excluding:Set<WorkflowNodeAddress>=[],allowGrowth:Bool=false)->WorkflowNodeAddress? {
        let geometry=CanvasGeometrySnapshot(registry:registry,presentation:canvasViewState.presentation,graph:graph,validation:validation)
        let p=geometry.projection.project(point)
        return inlineContainers.reversed().compactMap {container -> WorkflowNodeAddress? in
            guard container.isExpanded,let address=projection.addressByDisplayID[container.id],!excluding.contains(address),let node=graph.nodes.first(where:{$0.id==container.id}) else {return nil}
            let origin=geometry.projection.project(CGPoint(x:node.x,y:node.y))
            return CGRect(x:origin.x+15,y:origin.y+15,width:container.size.width-30+(allowGrowth ? 100:0),height:container.size.height-30+(allowGrowth ? 100:0)).contains(p) ? address:nil
        }.first
    }
}

extension WorkflowCanvasAdapter {
    /// True when handled (including an explicit refusal), false for the ordinary root path.
    func moveInlineNodes(ids:Set<String>,dx:Double,dy:Double)->Bool {
        guard !isBusy,!owner.hasPendingInspectorEdits,dx.isFinite,dy.isFinite else {return true}
        let addresses=Set(ids.compactMap {projection.addressByDisplayID[$0]})
        let selectedContainers=Set(inlineContainers.filter {ids.contains($0.id)}.compactMap {projection.addressByDisplayID[$0.id]})
        if !selectedContainers.isEmpty {
            let parents=Set(addresses.filter {address in
                guard let child=try? WorkflowInlineContainer.parse(in:owner.document,address:address) else {return true}
                return !selectedContainers.contains(child.root)
            })
            guard parents.allSatisfy({address in owner.document.nodes.contains(where:{$0.address==address})}) else {return false}
            _=apply(.move(addresses:parents,offset:.init(x:dx,y:dy)),lease:owner.lease());return true
        }
        guard addresses.count==1,let address=addresses.first,let display=projection.displayIDByAddress[address],let node=graph.nodes.first(where:{$0.id==display}) else {
            if addresses.contains(where:{(try? WorkflowInlineContainer.parse(in:owner.document,address:$0)) != nil}) {statusMessage="Move one container member at a time across its boundary.";return true}
            return false
        }
        let point=CGPoint(x:node.x+dx,y:node.y+dy),center=CGPoint(x:point.x+75,y:point.y+40)
        if let child=try? WorkflowInlineContainer.parse(in:owner.document,address:address),let root=owner.document.nodes.first(where:{$0.address==child.root}) {
            let target=inlineDropRoot(at:center,allowGrowth:true)
            if target==child.root {
                inlineState.positions[address.localID] = .init(x:point.x-root.x,y:point.y-root.y)
                inlineState.save(projectID:owner.projectID);refreshInlinePresentation();return true
            }
            if WorkflowMixtureContainer.contains(owner.document,root:child.root) {statusMessage="Keep expert layers inside their MoE container.";return true}
            guard target==nil else {statusMessage="Move this member onto the parent canvas before moving it into another container.";return true}
            let moved=WorkflowNodeAddress(domain:.model,localID:UUID().uuidString)
            if apply(.inlineContainer(root:child.root,command:.moveOut(childID:child.childID,address:moved,position:.init(x:point.x,y:point.y))),lease:owner.lease()) {
                inlineState.positions[address.localID]=nil;inlineState.save(projectID:owner.projectID)
                _=owner.select(.init(nodeAddresses:[moved]),lease:owner.lease())
            }
            return true
        }
        guard case .modelPrimitive=owner.authoringNode(at:address)?.payload,let target=inlineDropRoot(at:center,allowGrowth:true),let parent=owner.document.nodes.first(where:{$0.address==target}) else {return false}
        // An overlapping saved layout is not an instruction to change architecture.
        // Existing root nodes join only when this drag crosses into the container.
        if inlineDropRoot(at:CGPoint(x:node.x+75,y:node.y+40),allowGrowth:true)==target {return false}
        if apply(.inlineContainer(root:target,command:.moveIn(address:address,position:.init(x:point.x-parent.x,y:point.y-parent.y))),lease:owner.lease()) {
            let moved=WorkflowInlineContainer.address(root:target,childID:address.localID)
            inlineState.positions[moved.localID] = .init(x:point.x-parent.x,y:point.y-parent.y);inlineState.save(projectID:owner.projectID);refreshInlinePresentation()
            _=owner.select(.init(nodeAddresses:[moved]),lease:owner.lease())
        }
        return true
    }
}
