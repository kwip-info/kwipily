import AppKit
import SwiftUI

struct TensorCapturedGrid {
    struct Cell {
        let text:String
        let state:String
        let numericValue:Double?
    }
    let headers:[String]
    let gutterTitle:String
    let gutter:[String]
    let cells:[[Cell]]
    struct Selection:Equatable {let row:Int;let column:Int}
    func selected(_ selection:Selection?)->Cell? {
        guard let selection,cells.indices.contains(selection.row),cells[selection.row].indices.contains(selection.column) else {return nil}
        return cells[selection.row][selection.column]
    }
}

/// Uses the same native table chrome as captured tables. A synchronized gutter
/// stays fixed horizontally; native table headers stay fixed vertically.
struct TensorCapturedNativeGrid:NSViewRepresentable {
    let grid:TensorCapturedGrid
    let accent:Color
    let shading:Bool
    @Binding var selection:TensorCapturedGrid.Selection?
    func makeCoordinator()->Coordinator {Coordinator(self)}
    func makeNSView(context:Context)->TensorCapturedGridScrolls {
        let view=TensorCapturedGridScrolls(frame:.zero)
        view.mainTable.delegate=context.coordinator;view.mainTable.dataSource=context.coordinator
        view.gutterTable.delegate=context.coordinator;view.gutterTable.dataSource=context.coordinator
        view.mainTable.onSelect={row,column in context.coordinator.select(row:row,column:column)}
        view.mainTable.copyValue={context.coordinator.copySelection()}
        view.mainTable.setAccessibilityLabel("Captured tensor values")
        updateNSView(view,context:context)
        return view
    }
    func updateNSView(_ view:TensorCapturedGridScrolls,context:Context) {
        let coordinator=context.coordinator;coordinator.parent=self;coordinator.updating=true
        coordinator.shadingScale=grid.cells.flatMap {$0}.compactMap(\.numericValue).filter(\.isFinite).map {abs($0)}.max() ?? 0
        defer {coordinator.updating=false}
        if coordinator.headers != grid.headers || coordinator.gutterTitle != grid.gutterTitle {
            coordinator.headers=grid.headers;coordinator.gutterTitle=grid.gutterTitle
            view.mainTable.tableColumns.forEach {view.mainTable.removeTableColumn($0)}
            for (index,title) in grid.headers.enumerated() {
                let column=NSTableColumn(identifier:.init(String(index)))
                column.title=title;column.width=title == "Coordinate" ? 160:title == "State" ? 110:170;column.minWidth=70
                view.mainTable.addTableColumn(column)
            }
            view.gutterTable.tableColumns.forEach {view.gutterTable.removeTableColumn($0)}
            let column=NSTableColumn(identifier:.init("gutter"));column.title=grid.gutterTitle;column.width=70;column.minWidth=70
            view.gutterTable.addTableColumn(column)
        }
        view.mainTable.reloadData();view.gutterTable.reloadData()
        view.resizeDocuments()
        if let selection,grid.selected(selection) != nil {view.mainTable.selectRowIndexes(IndexSet(integer:selection.row),byExtendingSelection:false)}
        else {view.mainTable.deselectAll(nil)}
    }
    final class Coordinator:NSObject,NSTableViewDataSource,NSTableViewDelegate {
        var parent:TensorCapturedNativeGrid
        var headers:[String]=[]
        var gutterTitle=""
        var updating=false
        var shadingScale:Double=0
        init(_ parent:TensorCapturedNativeGrid) {self.parent=parent}
        func numberOfRows(in tableView:NSTableView)->Int {parent.grid.cells.count}
        func select(row:Int,column:Int) {
            guard !updating,parent.grid.cells.indices.contains(row),parent.grid.cells[row].indices.contains(column) else {return}
            parent.selection = .init(row:row,column:column)
        }
        func copySelection() {
            guard let cell=parent.grid.selected(parent.selection) else {return}
            NSPasteboard.general.clearContents();NSPasteboard.general.setString(cell.text,forType:.string)
        }
        func tableViewSelectionDidChange(_ notification:Notification) {
            guard !updating,let table=notification.object as? TensorCapturedCellTable,parent.grid.cells.indices.contains(table.selectedRow) else {return}
            select(row:table.selectedRow,column:parent.selection?.column ?? 0)
        }
        func tableView(_ tableView:NSTableView,viewFor tableColumn:NSTableColumn?,row:Int)->NSView? {
            guard parent.grid.cells.indices.contains(row),let id=tableColumn?.identifier.rawValue else {return nil}
            let cell=NSTableCellView();cell.wantsLayer=true
            let text:String
            var color=NSColor.labelColor
            if id == "gutter" {
                guard parent.grid.gutter.indices.contains(row) else {return nil}
                text=parent.grid.gutter[row] // Let AppKit paint the adaptive table background.
            } else {
                guard let column=Int(id),parent.grid.cells[row].indices.contains(column) else {return nil}
                let value=parent.grid.cells[row][column];text=value.text
                if value.state != "Captured" {color = .secondaryLabelColor}
                if parent.shading,let numeric=value.numericValue,numeric.isFinite {
                    let scale=shadingScale
                    if scale>0 {
                        let tint=numeric<0 ? NSColor.systemPurple:NSColor(parent.accent)
                        cell.layer?.backgroundColor=tint.withAlphaComponent(0.025+0.12*min(1,abs(numeric)/scale)).cgColor
                    }
                }
                if parent.selection == .init(row:row,column:column) {
                    cell.layer?.borderWidth=1;cell.layer?.borderColor=NSColor(parent.accent).cgColor
                }
            }
            let field=NSTextField(labelWithString:text)
            field.font = .monospacedSystemFont(ofSize:11,weight:.regular);field.textColor=color
            field.lineBreakMode = .byTruncatingTail;field.toolTip=text;field.translatesAutoresizingMaskIntoConstraints=false
            cell.addSubview(field);cell.textField=field
            NSLayoutConstraint.activate([field.leadingAnchor.constraint(equalTo:cell.leadingAnchor,constant:9),field.trailingAnchor.constraint(equalTo:cell.trailingAnchor,constant:-9),field.centerYAnchor.constraint(equalTo:cell.centerYAnchor)])
            cell.setAccessibilityLabel(text)
            return cell
        }
    }
}

final class TensorCapturedCellTable:NSTableView {
    var onSelect:((Int,Int)->Void)?
    var copyValue:(()->Void)?
    override func mouseDown(with event:NSEvent) {
        let point=convert(event.locationInWindow,from:nil),row=row(at:point),column=column(at:point)
        if row>=0,column>=0 {onSelect?(row,column)}
        super.mouseDown(with:event)
    }
    @objc func copy(_ sender:Any?) {copyValue?()}
}

final class TensorCapturedGridScrolls:NSView {
    let main=CapturedTableScrollView(frame:.zero)
    let gutter=CapturedTableScrollView(frame:.zero)
    let mainTable=TensorCapturedCellTable()
    var gutterTable:NSTableView {gutter.documentView as! NSTableView}
    private var synchronizing=false
    override init(frame:NSRect) {
        super.init(frame:frame)
        main.documentView=mainTable
        for table in [mainTable,gutterTable] {
            table.headerView=NSTableHeaderView();table.columnAutoresizingStyle = .noColumnAutoresizing
            table.rowHeight=27;table.intercellSpacing = .init(width:0,height:1)
            table.gridStyleMask=[.solidVerticalGridLineMask,.solidHorizontalGridLineMask]
            table.allowsColumnReordering=false;table.allowsMultipleSelection=false
            table.selectionHighlightStyle = .none;table.usesAlternatingRowBackgroundColors=false
        }
        gutter.hasVerticalScroller=false
        gutterTable.allowsEmptySelection=true
        gutterTable.setAccessibilityLabel("Tensor coordinate gutter")
        for scroll in [gutter,main] {
            scroll.translatesAutoresizingMaskIntoConstraints=false;addSubview(scroll)
            scroll.contentView.postsBoundsChangedNotifications=true
            NotificationCenter.default.addObserver(self,selector:#selector(scrolled(_:)),name:NSView.boundsDidChangeNotification,object:scroll.contentView)
        }
        NSLayoutConstraint.activate([gutter.leadingAnchor.constraint(equalTo:leadingAnchor),gutter.topAnchor.constraint(equalTo:topAnchor),gutter.bottomAnchor.constraint(equalTo:bottomAnchor),gutter.widthAnchor.constraint(equalToConstant:72),main.leadingAnchor.constraint(equalTo:gutter.trailingAnchor),main.trailingAnchor.constraint(equalTo:trailingAnchor),main.topAnchor.constraint(equalTo:topAnchor),main.bottomAnchor.constraint(equalTo:bottomAnchor)])
    }
    override func layout() {super.layout();resizeDocuments()}
    func resizeDocuments() {
        for (scroll,table) in [(main,mainTable),(gutter,gutterTable)] {
            let width=table.tableColumns.reduce(CGFloat(0)) {$0+$1.width+table.intercellSpacing.width}
            let height=CGFloat(table.numberOfRows)*(table.rowHeight+table.intercellSpacing.height)
            table.setFrameSize(.init(width:max(width,scroll.contentSize.width),height:max(height,scroll.contentSize.height)))
            scroll.contentView.setBoundsOrigin(scroll.contentView.constrainBoundsRect(scroll.contentView.bounds).origin)
        }
        // The gutter has no independent horizontal navigation.
        gutter.hasHorizontalScroller=false
        main.hasHorizontalScroller=mainTable.tableColumns.reduce(CGFloat(0)) {$0+$1.width} > main.contentSize.width+1
    }
    required init?(coder:NSCoder) {fatalError("init(coder:) has not been implemented")}
    deinit {NotificationCenter.default.removeObserver(self)}
    @objc private func scrolled(_ notification:Notification) {
        guard !synchronizing,let clip=notification.object as? NSClipView else {return}
        synchronizing=true;defer {synchronizing=false}
        let destination=clip === main.contentView ? gutter:main
        destination.contentView.scroll(to:.init(x:destination.contentView.bounds.minX,y:clip.bounds.minY))
        destination.reflectScrolledClipView(destination.contentView)
    }
}
