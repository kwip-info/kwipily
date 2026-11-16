import SwiftUI
import AppKit

/// Author tables are bounded separately from demanded runtime tables. Padding never shifts cells.
struct LessonVisualTable {
    let columns:[String]
    let rows:[[String]]
    let totalRows:Int
    let totalColumns:Int
    var isBounded:Bool {rows.count<totalRows || columns.count<totalColumns}
    init(columns:[String],rows:[[String]]) {
        let visibleColumns=Array(columns.prefix(32))
        self.columns=visibleColumns;totalRows=rows.count;totalColumns=columns.count
        self.rows=rows.prefix(64).map {row in (0..<visibleColumns.count).map {$0<row.count ? row[$0]:""}}
    }
}

@MainActor enum LessonBundledImage {
    private static let cache:NSCache<NSString,NSImage> = {let result=NSCache<NSString,NSImage>();result.countLimit=32;return result}()
    static func image(_ name:String)->NSImage? {
        guard !name.isEmpty,!name.contains("/"),!name.contains("\\"),!name.contains("..") else {return nil}
        if let value=cache.object(forKey:name as NSString) {return value}
        guard let url=Bundle.module.url(forResource:name,withExtension:nil) ?? Bundle.module.url(forResource:name,withExtension:nil,subdirectory:"ComponentIcons") ?? Bundle.module.url(forResource:name,withExtension:"png") ?? Bundle.module.url(forResource:name,withExtension:"png",subdirectory:"ComponentIcons"),
              let size=try? url.resourceValues(forKeys:[.fileSizeKey]).fileSize,size<=2_000_000,
              let image=NSImage(contentsOf:url) else {return nil}
        cache.setObject(image,forKey:name as NSString);return image
    }
}

struct LessonVisualView:View {
    @Environment(\.lessonTypography) private var typography
    let block:LessonContentBlock
    @Environment(\.lessonContentAccent) private var topicAccent
    @Environment(\.appearanceTheme) private var theme
    private var kind:LessonVisual.Kind? {block.visual?.kind ?? (!block.columns.isEmpty ? .table:block.imageName != nil ? .image:nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            switch kind {
            case .table,.matrix:table
            case .equation:
                ThemedScrollView(.horizontal) {Text(block.text).font(.system(size:typography.size(21),design:.serif)).padding(.vertical,12).textSelection(.enabled)}
                ForEach(Array((block.visual?.definitions ?? []).prefix(32).enumerated()),id:\.offset) {_,definition in
                    HStack(alignment:.top,spacing:12) {
                        Text(definition.symbol).font(.system(size:typography.size(12),weight:.semibold,design:.monospaced)).foregroundStyle(topicAccent).frame(minWidth:25,alignment:.leading)
                        LessonRichText(text:definition.meaning)
                    }
                    Divider()
                }
            case .diagram:diagram
            case .image:
                if let name=block.imageName,let image=LessonBundledImage.image(name) {
                    Image(nsImage:image).resizable().scaledToFit().frame(maxHeight:240)
                        .frame(maxWidth:.infinity).accessibilityLabel(block.alternativeText ?? block.title)
                } else {
                    Label("Lesson image unavailable",systemImage:"photo").foregroundStyle(MLTheme.muted)
                    if let alternative=block.alternativeText {LessonRichText(text:alternative)}
                }
            case .code:
                Text(AttributedString(PythonSyntax.attributed(block.text,palette:theme.native)))
                    .font(.system(size:typography.size(11),design:.monospaced)).textSelection(.enabled).fixedSize(horizontal:false,vertical:true)
            case nil:LessonRichText(text:block.text)
            }
            if let caption=block.visual?.caption,!caption.isEmpty {
                Divider();Text(caption).font(.system(size:typography.size(10))).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
            }
        }
    }
    private var table:some View {
        let table=LessonVisualTable(columns:block.columns,rows:block.rows)
        let widths=table.columns.enumerated().map {index,column in
            CGFloat(min(180,max(70,(max(column.count,table.rows.map {$0[index].count}.max() ?? 0))*Int(7 * typography.scale)+16)))
        }
        return VStack(alignment:.leading,spacing:8) {
            ThemedScrollView(.horizontal) {
                VStack(alignment:.leading,spacing:0) {
                    tableRow(table.columns,widths:widths,header:true)
                    ForEach(Array(table.rows.enumerated()),id:\.offset) {_,row in tableRow(row,widths:widths,header:false)}
                }.frame(width:widths.reduce(0,+),alignment:.leading)
            }
            if table.isBounded {Text("Showing \(table.rows.count) of \(table.totalRows) authored rows · \(table.columns.count) of \(table.totalColumns) columns").font(.system(size:typography.size(10))).foregroundStyle(MLTheme.muted)}
        }
    }
    private func tableRow(_ cells:[String],widths:[CGFloat],header:Bool)->some View {
        HStack(alignment:.top,spacing:0) {
            ForEach(Array(cells.enumerated()),id:\.offset) {index,value in
                Text(value).font(.system(size:typography.size(header ? 10:11),weight:header ? .semibold:.regular,design:header ? .default:.monospaced))
                    .foregroundStyle(header ? MLTheme.muted:Color.primary).lineLimit(3).help(value).textSelection(.enabled)
                    .padding(.horizontal,8).padding(.vertical,7).frame(width:widths[index],alignment:.leading)
            }
        }.background(header ? MLTheme.panel:MLTheme.background).overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    private var diagram:some View {
        GeometryReader {geometry in
            let nodes=Array((block.visual?.nodes ?? []).prefix(12))
            let layout=LessonDiagramLayout(count:nodes.count,availableWidth:geometry.size.width)
            let positions=Dictionary(nodes.enumerated().map {($0.element.id,$0.offset)},uniquingKeysWith:{first,_ in first})
            ThemedScrollView(.horizontal) {
                ZStack(alignment:.topLeading) {
                    Canvas {context,_ in
                        for link in (block.visual?.links ?? []).prefix(24) {
                            guard let from=positions[link.from],let to=positions[link.to],from != to else {continue}
                            let forward=to>from
                            let start=CGPoint(x:CGFloat(from)*layout.step+(forward ? layout.nodeWidth:0),y:14 + 32 * typography.scale)
                            let end=CGPoint(x:CGFloat(to)*layout.step+(forward ? 0:layout.nodeWidth),y:14 + 32 * typography.scale)
                            var path=Path();path.move(to:start)
                            if abs(to-from)>1 {path.addLine(to:CGPoint(x:start.x,y:10));path.addLine(to:CGPoint(x:end.x,y:10))}
                            path.addLine(to:end);context.stroke(path,with:.color(topicAccent),lineWidth:1.5)
                            var arrow=Path();arrow.move(to:CGPoint(x:end.x+(forward ? -5:5),y:end.y-4));arrow.addLine(to:end);arrow.addLine(to:CGPoint(x:end.x+(forward ? -5:5),y:end.y+4))
                            context.stroke(arrow,with:.color(topicAccent),lineWidth:1.5)
                        }
                    }.allowsHitTesting(false).accessibilityHidden(true)
                    ForEach(Array(nodes.enumerated()),id:\.offset) {index,node in
                        VStack(alignment:.leading,spacing:5) {
                            Text(node.title).font(.system(size:typography.size(11),weight:.semibold)).lineLimit(2)
                            if let subtitle=node.subtitle {Text(subtitle).font(.system(size:typography.size(10))).foregroundStyle(MLTheme.muted).lineLimit(2)}
                        }.padding(10).frame(width:layout.nodeWidth,height:64 * typography.scale,alignment:.leading).background(MLTheme.panel)
                            .clipShape(RoundedRectangle(cornerRadius:7)).overlay(RoundedRectangle(cornerRadius:7).stroke(MLTheme.rule))
                            .overlay {if node.emphasized == true {CanvasLessonFocusHalo(light:theme.colorScheme == .light)}}
                            .offset(x:CGFloat(index)*layout.step,y:14)
                    }
                }.frame(width:layout.width,height:28 + 64 * typography.scale).padding(5)
            }.accessibilityElement(children:.combine)
        }.frame(height:38 + 64 * typography.scale)
    }
}

struct LessonDiagramLayout {
    let nodeWidth:CGFloat
    let width:CGFloat
    var step:CGFloat {nodeWidth+24}
    init(count:Int,availableWidth:CGFloat) {
        nodeWidth=count == 2 ? min(146,max(92,(availableWidth-34)/2)):146
        width=max(146,CGFloat(min(12,max(0,count)))*(nodeWidth+24)-24)
    }
}
