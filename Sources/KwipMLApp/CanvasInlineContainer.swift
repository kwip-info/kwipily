import SwiftUI
import KwipMLCore

/// Presentation only. Children remain ordinary nodes on the shared canvas.
/// Size and anchors use unscaled, projected canvas points; node positions remain model coordinates.
struct CanvasInlineContainer:Equatable,Identifiable {
    struct BoundaryLink:Equatable {
        let containerPort:String
        let childNodeID:String
        let childPort:String
        /// Input joins the container input to a child input; output joins two outputs.
        let direction:PortDirection
        /// A routed operation binds its chain boundary, not an eager child tensor port.
        var routed:Bool=false
        var containerNodeID:String?=nil
    }
    struct Lane:Equatable {let id:String;let title:String;let y:CGFloat}
    var lanes:[Lane]=[]
    let id:String
    var isExpanded:Bool
    var size:CGSize
    var childNodeIDs:[String]
    var boundaryLinks:[BoundaryLink]
    var inputOffsets:[String:CGPoint]
    var outputOffsets:[String:CGPoint]
    var hasRoutedBindings:Bool {boundaryLinks.contains(where:{$0.routed})}
    init(id:String,isExpanded:Bool,size:CGSize,childNodeIDs:[String],boundaryLinks:[BoundaryLink]=[],inputOffsets:[String:CGPoint]=[:],outputOffsets:[String:CGPoint]=[:]) {
        self.id=id;self.isExpanded=isExpanded;self.size=size;self.childNodeIDs=childNodeIDs
        self.boundaryLinks=boundaryLinks;self.inputOffsets=inputOffsets;self.outputOffsets=outputOffsets
    }
}

struct CanvasInlineContainerFrame:View {
    let size:CGSize
    let tint:Color
    let selected:Bool
    let targeted:Bool
    var body:some View {
        RoundedRectangle(cornerRadius:8).fill(tint.opacity(targeted ? 0.12:0.045))
            .overlay(RoundedRectangle(cornerRadius:8).stroke(tint.opacity(selected || targeted ? 1:0.65),lineWidth:targeted ? 2:1))
            .overlay(alignment:.top) {tint.opacity(0.10).frame(height:CanvasNodeGeometry.headerHeight)}
            .overlay(alignment:.top) {Rectangle().fill(tint.opacity(0.35)).frame(height:1).offset(y:CanvasNodeGeometry.headerHeight)}
            .frame(width:size.width,height:size.height)
            .allowsHitTesting(false).accessibilityHidden(true)
    }
}

struct CanvasInlineContainerHeader:View {
    let title:String
    let typeTitle:String
    let iconType:String
    let count:Int
    let expanded:Bool
    let tint:Color
    var fit:()->Void={}
    let toggle:()->Void
    var body:some View {
        HStack(spacing:9) {
            Button(action:toggle) {Image(systemName:expanded ? "chevron.down":"chevron.right").font(.system(size:11,weight:.semibold)).frame(width:18,height:28)}
                .buttonStyle(.plain).accessibilityLabel(expanded ? "Collapse container":"Expand container")
            ComponentIcon(type:iconType,size:25)
            VStack(alignment:.leading,spacing:3) {
                Text(title).font(.system(size:12,weight:.semibold)).lineLimit(1)
                Text("\(typeTitle) · \(count) components").font(.system(size:10)).foregroundStyle(MLTheme.muted).lineLimit(1)
            }
            Spacer(minLength:4)
            if expanded {Button(action:fit) {Image(systemName:"arrow.down.right.and.arrow.up.left").font(.system(size:11)).frame(width:24,height:24)}.buttonStyle(.plain).accessibilityLabel("Fit container to components")}
        }.padding(.horizontal,10).frame(height:CanvasNodeGeometry.headerHeight)
            .background(MLTheme.panel.opacity(0.96))
            .overlay(alignment:.bottom) {Rectangle().fill(tint.opacity(0.35)).frame(height:1)}
    }
}

/// One visual boundary per direction, with independent named ports; never worker nodes.
struct CanvasInlineBoundaryGroup:View {
    let title:String
    let ports:[String]
    let output:Bool
    let tint:Color
    var body:some View {
        VStack(spacing:0) {
            Text(title).font(.system(size:11,weight:.semibold)).foregroundStyle(tint)
                .frame(maxWidth:.infinity,alignment:output ? .trailing:.leading).padding(.horizontal,14).frame(height:24)
                .background(tint.opacity(0.10))
            ForEach(ports,id:\.self) {port in
                Text(port).font(.system(size:10,weight:.medium)).foregroundStyle(MLTheme.muted)
                    .lineLimit(1).frame(maxWidth:.infinity,alignment:output ? .trailing:.leading)
                    .padding(.horizontal,14).frame(height:58)
                    .overlay(alignment:.bottom) {Rectangle().fill(tint.opacity(0.15)).frame(height:1)}
            }
            if ports.isEmpty {Text("No ports").font(.system(size:10)).foregroundStyle(MLTheme.muted).frame(height:36)}
        }.frame(width:150).background(MLTheme.panel.opacity(0.97))
            .overlay(RoundedRectangle(cornerRadius:6).stroke(tint.opacity(0.45)))
            .clipShape(RoundedRectangle(cornerRadius:6)).allowsHitTesting(false).accessibilityHidden(true)
    }
}

struct CanvasInlineLaneLabels:View {
    let lanes:[CanvasInlineContainer.Lane]
    let tint:Color
    var body:some View {
        ForEach(lanes,id:\.id) {lane in
            Text(lane.title).font(.system(size:12,weight:.semibold)).foregroundStyle(tint)
                .offset(x:200,y:lane.y).allowsHitTesting(false)
        }
    }
}
