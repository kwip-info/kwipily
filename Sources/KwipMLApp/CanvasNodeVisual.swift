import SwiftUI
import KwipMLCore

struct CanvasTablePortRow: Equatable, Identifiable {
    let port:String
    let output:Bool
    let anchorY:CGFloat
    let shortLabel:String
    let description:String
    var id:String {(output ? "output:":"input:")+port}
}

/// Immutable visible content. Camera motion and unrelated node edits need not
/// rebuild this node's text/artwork/layout. Gestures stay on the live canvas wrapper.
/// Synthesized equality includes every visual input; no callbacks or editor state are retained.
struct CanvasNodeVisual: View, Equatable {
    let name:String
    let title:String
    let iconType:String
    let nodeSize:CGSize
    let selected:Bool
    let familyTint:Color
    let selectionTint:Color
    let light:Bool
    let feedback:CanvasNodeFeedback
    let flowStatus:String
    let rows:[CanvasTablePortRow]
    private var style:CanvasNodeSurfaceStyle {.init(detail:.medium)}
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                ComponentIcon(type:iconType,size:style.iconSize)
                VStack(alignment:.leading,spacing:3) {
                    if !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && name != title {
                        Text(name).font(.system(size:12,weight:.semibold)).lineLimit(1).help(name)
                        Text(title).font(.system(size:10)).foregroundStyle(MLTheme.muted).lineLimit(1)
                    } else {
                        Text(title).font(.system(size:12,weight:.semibold)).lineLimit(1).help(title)
                    }
                }
                Spacer(minLength:0)
            }
            .padding(.horizontal,12)
            .frame(height:CanvasNodeGeometry.headerHeight)
            .background(familyTint.opacity(light ? 0.08 : 0.06))
            // Separators and row labels use the same geometry as hit testing and wires.
            ZStack(alignment:.topLeading) {
                ForEach(rows) { row in portLabel(row) }
                if rows.isEmpty {
                    Text("No connection ports").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                        .padding(.horizontal,12).frame(height:CanvasNodeGeometry.rowHeight)
                }
            }
            .frame(height:nodeSize.height-CanvasNodeGeometry.headerHeight-CanvasNodeGeometry.footerHeight)
            HStack(spacing:5) {
                Image(systemName:feedback.executionFailure != nil ? "exclamationmark.circle.fill" : feedback.isMisconfigured ? "exclamationmark.triangle" : flowStatusSymbol(flowStatus))
                Text(feedback.executionFailure != nil ? "Run failed" : feedback.isMisconfigured ? "Review configuration" : flowStatus).lineLimit(1)
                Spacer(minLength:0)
            }
            .font(.system(size:9,weight:.medium))
            .foregroundStyle(feedback.executionFailure != nil ? Color.red : feedback.isMisconfigured ? Color.orange : MLTheme.muted)
            .padding(.horizontal,12).frame(height:CanvasNodeGeometry.footerHeight)
            .overlay(alignment:.top) {Rectangle().fill(familyTint.opacity(0.3)).frame(height:1)}
        }
        .frame(width:nodeSize.width,height:nodeSize.height,alignment:.topLeading)
        .background(light ? Color.white : MLTheme.panel)
        .clipShape(RoundedRectangle(cornerRadius:style.cornerRadius))
        .overlay(RoundedRectangle(cornerRadius:style.cornerRadius).stroke(selected ? selectionTint : familyTint.opacity(light ? 0.9 : 0.65),lineWidth:style.borderWidth(isSelected:selected)))
        .overlay(CanvasNodeFeedbackOverlay(feedback:feedback))
    }
    private func portLabel(_ row:CanvasTablePortRow)->some View {
        VStack(alignment:row.output ? .trailing:.leading,spacing:2) {
            Text(iconType == "kwip.data.union" && !row.output ? UnionConfiguration.title(row.port):WorkflowPortPresentation.title(type:iconType,port:row.port,direction:row.output ? .output:.input)).font(.system(size:11,weight:.medium)).lineLimit(1)
            Text(row.shortLabel).font(.system(size:9)).foregroundStyle(MLTheme.muted).lineLimit(1)
        }
        .frame(width:nodeSize.width-64,height:CanvasNodeGeometry.rowHeight,alignment:row.output ? .trailing:.leading)
        .frame(width:nodeSize.width,height:CanvasNodeGeometry.rowHeight)
        .overlay(alignment:.top) {Rectangle().fill(familyTint.opacity(0.3)).frame(height:1)}
        .position(x:nodeSize.width/2,y:row.anchorY-CanvasNodeGeometry.headerHeight)
        .help("\(row.output ? "Output":"Input") \(row.port). \(row.description)")
        .allowsHitTesting(false)
    }
    private func flowStatusSymbol(_ status:String)->String {
        switch status {
        case "Cached", "Reused": return "archivebox"
        case "Computed": return "checkmark.circle"
        case "Stale": return "clock.arrow.circlepath"
        default: return "circle.dotted"
        }
    }
}
