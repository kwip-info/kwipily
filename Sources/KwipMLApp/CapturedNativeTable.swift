import AppKit
import SwiftUI
import KwipMLCore

/// A bounded sample viewport. AppKit keeps the header aligned during horizontal
/// scrolling and pinned during vertical scrolling, with draggable scrollbars.
final class CapturedTableScrollView: NSScrollView {
    override init(frame: NSRect) {
        super.init(frame: frame)
        clipsToBounds = true
        hasHorizontalScroller = true
        hasVerticalScroller = true
        autohidesScrollers = false
        scrollerStyle = .legacy
        borderType = .noBorder
        let table = NSTableView()
        table.headerView = NSTableHeaderView()
        // Flat grid rows and custom selection must share identical content bounds.
        // Automatic AppKit inset rows leave a rounded trailing gutter under a full-width highlight.
        table.style = .plain
        table.columnAutoresizingStyle = .noColumnAutoresizing
        table.rowHeight = 30
        table.intercellSpacing = NSSize(width: 0, height: 1)
        table.usesAlternatingRowBackgroundColors = true
        table.gridStyleMask = [.solidVerticalGridLineMask, .solidHorizontalGridLineMask]
        table.allowsMultipleSelection = false
        table.allowsColumnReordering = false
        documentView = table
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}

/// One bounded portion of captured scalar text. Never a file/source reader.
struct CapturedCellPage {
    static let characters=16_384
    let text:String
    let start:Int
    let total:Int
    var message:String? = nil
    static let loading=Self(text:"",start:0,total:0,message:"Loading captured text page…")
    static func unavailable(_ reason:String)->Self {Self(text:"",start:0,total:0,message:"Text preview unavailable: "+reason)}
    static func slice(_ text:String,start:Int)->Self {
        let scalars=text.unicodeScalars
        return Self(text:String(String.UnicodeScalarView(scalars.dropFirst(start).prefix(characters))),start:start,total:scalars.count)
    }
    var previous:Int? {message==nil && start>0 ? max(0,start-Self.characters):nil}
    var next:Int? {message==nil && start+Self.characters<total ? start+Self.characters:nil}
    var coverage:String {message ?? (total>0 ? "Characters \(start+1)–\(min(start+Self.characters,total)) of \(total) · Read-only captured text":"Read-only captured value")}
}

private struct CapturedCellToolbar:View {
    let title:String,page:CapturedCellPage
    let copy:()->Void,close:()->Void,move:(Int)->Void
    var body:some View {
        HStack(spacing:8) {
            VStack(alignment:.leading,spacing:4) {
                Text(title).font(.system(size:12,weight:.semibold)).lineLimit(1).truncationMode(.middle)
                Text(page.coverage).font(.system(size:10)).foregroundStyle(MLTheme.muted).lineLimit(1).help(page.coverage)
            }
            Spacer(minLength:4)
            if page.previous != nil || page.next != nil {
                Button {if let value=page.previous {move(value)}} label:{Image(systemName:"chevron.left")}.disabled(page.previous==nil).accessibilityLabel("Previous text page")
                Button {if let value=page.next {move(value)}} label:{Image(systemName:"chevron.right")}.disabled(page.next==nil).accessibilityLabel("Next text page")
            }
            Button(page.next != nil || page.previous != nil ? "Copy page":"Copy",action:copy).disabled(page.message != nil)
            Button("Close",action:close)
        }.buttonStyle(FlatButtonStyle()).padding(8)
    }
}

final class CapturedCellTextView:NSTextView {
    var onClose:(()->Void)?
    override func cancelOperation(_ sender:Any?){onClose?()}
}
final class CapturedTableContainerView:NSView {
    override var isFlipped:Bool {true}
    let scroll=CapturedTableScrollView()
    let detail=NSView(),title=NSTextField(labelWithString:"")
    let text=CapturedCellTextView(),body=NSScrollView()
    private let toolbar=NSHostingView(rootView:AnyView(EmptyView()))
    var theme=EnvironmentValues().appearanceTheme {didSet {applyTheme();refreshToolbar()}}
    var onClose:(()->Void)?,onPage:((Int)->Void)?
    private(set) var page=CapturedCellPage.loading
    override init(frame:NSRect) {
        super.init(frame:frame)
        // Native table layers may temporarily retain their previous split-pane
        // frame during SwiftUI layout. They must never paint across siblings.
        clipsToBounds=true;wantsLayer=true;layer?.masksToBounds=true
        detail.clipsToBounds=true
        addSubview(detail);addSubview(scroll);detail.addSubview(toolbar);detail.addSubview(body)
        detail.wantsLayer=true;detail.layer?.borderWidth=1;detail.layer?.cornerRadius=0
        body.hasVerticalScroller=true;body.autohidesScrollers=false;body.documentView=text
        body.borderType = .noBorder
        text.isEditable=false;text.isSelectable=true;text.isRichText=false;text.font = .monospacedSystemFont(ofSize:12,weight:.regular)
        text.isHorizontallyResizable=false;text.isVerticallyResizable=true;text.autoresizingMask = [.width]
        text.textContainer?.widthTracksTextView=true;text.textContainerInset=NSSize(width:8,height:8)
        text.setAccessibilityLabel("Captured cell contents");text.onClose={ [weak self] in self?.closeDetail() }
        detail.isHidden=true;applyTheme()
    }
    required init?(coder:NSCoder){fatalError("init(coder:) has not been implemented")}
    override func layout() {
        super.layout()
        let height=detail.isHidden ? 0:min(210,max(90,bounds.height*0.55),max(0,bounds.height-30))
        detail.frame=NSRect(x:0,y:0,width:bounds.width,height:height)
        toolbar.frame=NSRect(x:1,y:max(0,height-48),width:max(0,bounds.width-2),height:48)
        // detail uses AppKit's bottom-left coordinates; text stays below its header.
        body.frame=NSRect(x:8,y:8,width:max(0,bounds.width-16),height:max(0,height-56))
        let gap:CGFloat=detail.isHidden ? 0:8
        scroll.frame=NSRect(x:0,y:height+gap,width:bounds.width,height:max(0,bounds.height-height-gap))
    }
    private func applyTheme(){detail.layer?.borderColor=theme.native.rule.cgColor;detail.layer?.backgroundColor=theme.native.panel.cgColor;text.textColor=theme.native.text;text.backgroundColor=theme.native.background;body.backgroundColor=theme.native.background}
    private func refreshToolbar(){toolbar.rootView=AnyView(CapturedCellToolbar(title:title.stringValue,page:page,copy:{[weak self] in self?.copyValue()},close:{[weak self] in self?.closeDetail()},move:{[weak self] value in self?.onPage?(value)}).environment(\.appearanceTheme,theme).environment(\.colorScheme,theme.colorScheme))}
    func show(title:String,page:CapturedCellPage,focus:Bool=true){self.title.stringValue=title;self.page=page;text.string=page.text;text.scrollRangeToVisible(NSRange(location:0,length:0));detail.isHidden=false;refreshToolbar();needsLayout=true;layoutSubtreeIfNeeded();if focus {window?.makeFirstResponder(text)}}
    @objc func closeDetail(){if window?.firstResponder === text {window?.makeFirstResponder(scroll.documentView)};detail.isHidden=true;text.string="";page = .loading;onPage=nil;needsLayout=true;onClose?()}
    private func copyValue(){NSPasteboard.general.clearContents();NSPasteboard.general.setString(text.string,forType:.string)}
}

/// Shared table selection uses the configured application accent, in both key
/// and inactive windows; AppKit's default blue is never drawn underneath it.
final class CapturedThemedRowView:NSTableRowView {
    var selectionAccent=NSColor.systemPurple {didSet {needsDisplay=true}}
    override var interiorBackgroundStyle:NSView.BackgroundStyle {.normal}
    override func drawSelection(in dirtyRect:NSRect) {
        guard isSelected else {return}
        let content=bounds.intersection(visibleRect)
        guard !content.isEmpty else {return}
        NSGraphicsContext.saveGraphicsState();defer {NSGraphicsContext.restoreGraphicsState()}
        NSBezierPath(rect:content).addClip()
        selectionAccent.withAlphaComponent(0.16).setFill();content.fill()
        selectionAccent.setFill();NSRect(x:bounds.minX,y:bounds.minY,width:3,height:bounds.height).fill()
    }
}

final class CapturedSequenceOpenButton:NSButton {var ids:[Int64]=[];var columnName="";var capturedRow:CapturedDataGrid.Row?}

final class CapturedMediaOpenButton:NSButton {var reference:MediaArtifactReference?;var capturedRow:CapturedDataGrid.Row?}

final class CapturedImageOpenButton: NSButton { var reference: FileImageReference? }

struct CapturedNativeTable: NSViewRepresentable {
    @Environment(\.appearanceTheme) private var theme
    let columns: [CapturedDataGrid.Column]
    let rows: [CapturedDataGrid.Row]
    @Binding var selectedRow: Int?
    var rowHeaderTitle: String = "Sample"
    var fillsWidth=false
    var sort: TableGridSort = .init()
    var onSort: ((Int?) -> Void)? = nil
    var tokenVocabulary:TokenVocabularyResolver? = nil
    var imageResolver: FileImageArtifactResolver? = nil
    var onOpenImage: ((FileImageReference) -> Void)? = nil
    var onOpenMedia:((MediaArtifactReference)->Void)? = nil
    var onOpenMediaRow:((CapturedDataGrid.Row,MediaArtifactReference)->Void)? = nil
    var onOpenSequence: ((CapturedDataGrid.Row,String)->Bool)? = nil
    var rowIndexOffset:Int=1
    var onInspectCell:((CapturedDataGrid.Row,Int,Int,@escaping(CapturedCellPage)->Void)->Void)? = nil
    var onCloseCell:(()->Void)? = nil

    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeNSView(context: Context) -> CapturedTableContainerView {
        let container=CapturedTableContainerView(frame:.zero)
        let scroll=container.scroll
        context.coordinator.container=container
        container.onClose={ [weak coordinator=context.coordinator] in coordinator?.detailToken=nil;coordinator?.inspected=nil;coordinator?.parent.onCloseCell?() }
        let table = scroll.documentView as! NSTableView
        table.columnAutoresizingStyle=fillsWidth ? .uniformColumnAutoresizingStyle:.noColumnAutoresizing
        context.coordinator.selectionAccent=theme.native.accent
        table.delegate = context.coordinator
        table.dataSource = context.coordinator
        table.target=context.coordinator;table.doubleAction = #selector(Coordinator.inspectClickedCell(_:))
        table.setAccessibilityLabel(rowHeaderTitle == "Sample" ? "Captured table sample" : "Recorded table rows")
        updateNSView(container, context: context)
        return container
    }
    func updateNSView(_ container: CapturedTableContainerView, context: Context) {
        let scroll=container.scroll
        let coordinator = context.coordinator
        coordinator.parent = self
        container.theme=theme
        if let inspected=coordinator.inspected,!rows.contains(where:{$0.index==inspected.row && $0.cells.indices.contains(inspected.column) && $0.cells[inspected.column].text==inspected.text}),!container.detail.isHidden {container.closeDetail()}
        coordinator.selectionAccent=theme.native.accent
        let table = scroll.documentView as! NSTableView
        table.enumerateAvailableRowViews {row,_ in (row as? CapturedThemedRowView)?.selectionAccent=theme.native.accent}
        let identity = columns.map { [$0.name, $0.label, $0.dtype, $0.nullability] }
        if coordinator.columnIdentity != identity {
            coordinator.columnIdentity = identity
            table.tableColumns.forEach { table.removeTableColumn($0) }
            let number = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("sample"))
            number.title = rowHeaderTitle; number.width = 58; number.minWidth = 58; number.maxWidth=58
            table.addTableColumn(number)
            for (index, column) in columns.enumerated() {
                let native = NSTableColumn(identifier: NSUserInterfaceItemIdentifier(String(index)))
                native.title = column.label
                native.headerToolTip = "\(column.name) · \(column.dtype) · \(column.nullability)"
                native.width = column.dtype == "image" ? 360 : 150; native.minWidth = column.dtype == "image" ? 180 : 80
                table.addTableColumn(native)
            }
        }
        for (index, native) in table.tableColumns.enumerated() {
            let column: Int? = index == 0 ? nil : index - 1
            let title = index == 0 ? rowHeaderTitle : columns[index - 1].label
            native.title = title + (onSort != nil && (column == nil || columns[column!].dtype != "image") && sort.column == column ? (sort.descending ? " ↓" : " ↑") : "")
        }
        coordinator.updating = true
        table.reloadData()
        if let selectedRow, let index = rows.firstIndex(where: { $0.index == selectedRow }) {
            table.selectRowIndexes(IndexSet(integer: index), byExtendingSelection: false)
        } else { table.deselectAll(nil) }
        coordinator.updating = false
    }
    static func dismantleNSView(_ view:CapturedTableContainerView,coordinator:Coordinator){view.closeDetail()}
    final class Coordinator: NSObject, NSTableViewDataSource, NSTableViewDelegate {
        var parent: CapturedNativeTable
        var columnIdentity: [[String]] = []
        var updating = false
        init(_ parent: CapturedNativeTable) { self.parent = parent }
        var tokenWindow:NSWindowController?
        weak var container:CapturedTableContainerView?
        var detailToken:UUID?
        var inspected:(row:Int,column:Int,text:String)?
        @objc func inspectClickedCell(_ table:NSTableView) {
            inspect(row:table.clickedRow,column:table.clickedColumn-1,table:table)
        }
        func inspect(row:Int,column:Int,table:NSTableView) {
            guard parent.rows.indices.contains(row),parent.columns.indices.contains(column),parent.rows[row].cells.indices.contains(column) else {return}
            let value=parent.rows[row].cells[column]
            guard value.image==nil,value.media==nil,value.sequence==nil else {return}
            let field=(table.view(atColumn:column+1,row:row,makeIfNecessary:false) as? NSTableCellView)?.textField
            let measured=field?.bounds.width ?? 0
            let width=measured>0 ? measured:max(0,table.tableColumns[column+1].width-16)
            let font=field?.font ?? NSFont.monospacedSystemFont(ofSize:11,weight:.regular)
            guard value.text.contains("\n") || value.text.contains("\r") || (value.text as NSString).size(withAttributes:[.font:font]).width>width else {return}
            parent.onCloseCell?()
            let token=UUID();detailToken=token;inspected=(parent.rows[row].index,column,value.text)
            let title=parent.columns[column].label+" · Row \(parent.rows[row].index+parent.rowIndexOffset)"
            container?.onPage={ [weak self] start in self?.loadCell(row:row,column:column,start:start,title:title,token:token,focus:false) }
            loadCell(row:row,column:column,start:0,title:title,token:token,focus:true)
        }
        private func loadCell(row:Int,column:Int,start:Int,title:String,token:UUID,focus:Bool) {
            guard detailToken==token,parent.rows.indices.contains(row),parent.rows[row].cells.indices.contains(column) else {return}
            container?.show(title:title,page:parent.onInspectCell == nil ? .slice(parent.rows[row].cells[column].text,start:start):.loading,focus:focus)
            parent.onInspectCell?(parent.rows[row],column,start,{ [weak self] page in
                guard let self,self.detailToken==token else {return}
                self.container?.show(title:title,page:page,focus:false)
            })
        }

        func numberOfRows(in tableView: NSTableView) -> Int { parent.rows.count }
        func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
            guard parent.rows.indices.contains(row), let id = tableColumn?.identifier.rawValue else { return nil }
            let value: String
            var absent = false
            var sequence:[Int64]?
            var media:MediaArtifactReference?
            var image: FileImageReference?
            if id == "sample" { value = String(parent.rows[row].index + parent.rowIndexOffset) }
            else if let column = Int(id), parent.rows[row].cells.indices.contains(column) {
                let cell = parent.rows[row].cells[column]; value = cell.text; absent = cell.isAbsent; image = cell.image; media=cell.media; sequence=cell.sequence
            } else { return nil }
            let cell = NSTableCellView()
            cell.wantsLayer=true;cell.layer?.masksToBounds=true
            let field = NSTextField(labelWithString: value.replacingOccurrences(of:"\r\n",with:" ↵ ").replacingOccurrences(of:"\n",with:" ↵ ").replacingOccurrences(of:"\r",with:" ↵ "))
            field.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
            field.textColor = absent ? .secondaryLabelColor : .labelColor
            field.lineBreakMode = .byTruncatingTail
            field.maximumNumberOfLines=1
            field.cell?.wraps=false;field.cell?.isScrollable=false
            field.toolTip = value
            field.translatesAutoresizingMaskIntoConstraints = false
            cell.addSubview(field); cell.textField = field
            NSLayoutConstraint.activate([
                field.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 8),
                field.trailingAnchor.constraint(equalTo: cell.trailingAnchor, constant: image == nil && media == nil && sequence == nil ? -8 : -92),
                field.centerYAnchor.constraint(equalTo: cell.centerYAnchor)
            ])
            if let sequence {
                let button=CapturedSequenceOpenButton(title:"Inspect",target:self,action:#selector(openSequence(_:)))
                button.ids=sequence;button.capturedRow=parent.rows[row];button.columnName=parent.columns[Int(id)!].name;button.bezelStyle = .rounded;button.translatesAutoresizingMaskIntoConstraints=false
                cell.addSubview(button)
                NSLayoutConstraint.activate([button.trailingAnchor.constraint(equalTo:cell.trailingAnchor,constant:-5),button.centerYAnchor.constraint(equalTo:cell.centerYAnchor),button.widthAnchor.constraint(equalToConstant:82)])
            }
            if let media {
                let button=CapturedMediaOpenButton(title:"Open",target:self,action:#selector(openMedia(_:)))
                button.capturedRow=parent.rows[row];button.reference=media;button.bezelStyle = .rounded;button.isEnabled=parent.onOpenMedia != nil && parent.imageResolver?.contains(media)==true
                if !button.isEnabled {button.title="Unavailable"}
                button.translatesAutoresizingMaskIntoConstraints=false;cell.addSubview(button)
                NSLayoutConstraint.activate([button.trailingAnchor.constraint(equalTo:cell.trailingAnchor,constant:-5),button.centerYAnchor.constraint(equalTo:cell.centerYAnchor),button.widthAnchor.constraint(equalToConstant:82)])
            }
            if let image {
                let button = CapturedImageOpenButton(title: "Open", target: self, action: #selector(openImage(_:)))
                button.reference = image; button.bezelStyle = .rounded
                button.isEnabled = parent.onOpenImage != nil && parent.imageResolver?.contains(image) == true
                if !button.isEnabled { button.title = "Unavailable" }
                button.toolTip = button.isEnabled ? "Open verified captured image pixels" : "Captured image artifact is unavailable in this evidence surface"
                button.translatesAutoresizingMaskIntoConstraints = false; cell.addSubview(button)
                NSLayoutConstraint.activate([button.trailingAnchor.constraint(equalTo: cell.trailingAnchor, constant: -5),
                    button.centerYAnchor.constraint(equalTo: cell.centerYAnchor), button.widthAnchor.constraint(equalToConstant: 82)])
            }
            return cell
        }
        @objc private func openSequence(_ sender:CapturedSequenceOpenButton) {
            if let row=sender.capturedRow,parent.onOpenSequence?(row,sender.columnName)==true {return}
            let controller=NSHostingController(rootView:TokenSequenceViewer(ids:sender.ids,vocabulary:try? parent.tokenVocabulary?.resolve(column:sender.columnName)).environment(\.appearanceTheme,parent.theme).environment(\.colorScheme,parent.theme.colorScheme).tint(parent.theme.accentColor))
            let window=NSWindow(contentViewController:controller);window.title="Captured token sequence";window.setContentSize(NSSize(width:480,height:440))
            tokenWindow=NSWindowController(window:window);tokenWindow?.showWindow(nil)
        }
        @objc private func openMedia(_ sender:CapturedMediaOpenButton) {
            guard let reference=sender.reference,parent.imageResolver?.contains(reference)==true else {return}
            if let row=sender.capturedRow,let open=parent.onOpenMediaRow {open(row,reference)} else {parent.onOpenMedia?(reference)}
        }
        @objc private func openImage(_ sender: CapturedImageOpenButton) {
            guard let reference = sender.reference, parent.imageResolver?.contains(reference) == true else { return }
            parent.onOpenImage?(reference)
        }
        var selectionAccent=NSColor.systemPurple
        func tableView(_ tableView:NSTableView,rowViewForRow row:Int)->NSTableRowView? {
            let view=CapturedThemedRowView();view.selectionAccent=selectionAccent;return view
        }
        func tableView(_ tableView: NSTableView, didClick tableColumn: NSTableColumn) {
            guard !updating else { return }
            let index = Int(tableColumn.identifier.rawValue)
            if let index, parent.columns.indices.contains(index), ["image","audio","tensor","int64Sequence"].contains(parent.columns[index].dtype) { return }
            parent.onSort?(tableColumn.identifier.rawValue == "sample" ? nil : index)
        }
        func tableViewSelectionDidChange(_ notification: Notification) {
            guard !updating, let table = notification.object as? NSTableView else { return }
            parent.selectedRow = parent.rows.indices.contains(table.selectedRow) ? parent.rows[table.selectedRow].index : nil
        }
    }
}
