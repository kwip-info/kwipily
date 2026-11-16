import SwiftUI
import AppKit
import KwipMLCore

struct ComponentRibbonEntry: Identifiable, Equatable {
    let id:String
    let title:String
    let shortTitle:String
    let purpose:String
    let category:String
    let typeIcon:String
}

enum ComponentRibbonCatalog {
    enum Scope: Equatable { case common, all, category(String) }
    struct Group: Identifiable {
        let category: String
        let entries: [ComponentRibbonEntry]
        var id: String { category }
    }
    /// Curated defaults, not personalized usage or inferred availability.
    private static let commonPriority = [
        "kwip.data.sourceCSV", "kwip.data.sourceFiles", "kwip.data.columns", "kwip.data.formula", "kwip.data.textEncode", "kwip.data.sequenceWindows", "kwip.data.filterRows", "kwip.data.partition",
        "kwip.data.preprocess", "kwip.data.view", "input", "kwip.linear", "kwip.activation",
        "concat", "parallel_features", "kwip.block", "kwip.reshape", "output", "workflow.modelDefinition",
        "workflow.train", "workflow.saveModel", "workflow.frozenModel", "workflow.evaluateTest", "workflow.generateText"
    ]
    /// Source → preparation → partition → training-fitted processing → inspection/output.
    /// Presentation only: registry order and saved graphs retain their own authority.
    private static let dataPriority = [
        "kwip.data.sourceCSV", "kwip.data.sourceFiles", "kwip.data.columns", "kwip.data.formula", "kwip.data.textEncode", "kwip.data.sequenceWindows",
        "kwip.data.filterRows", "kwip.data.sort", "kwip.data.unique", "kwip.data.join", "kwip.data.union", "kwip.data.aggregate", "kwip.data.reshapeTable", "kwip.data.splitText", "kwip.data.missing",
        "kwip.data.sampleRows", "kwip.data.partition", "kwip.data.preprocess", "kwip.data.imputerFit",
        "kwip.data.oneHotEncoderFit", "kwip.data.standardScalerFit", "kwip.data.transform",
        "kwip.data.view", "kwip.data.outputDelimited", "output"
    ]
    // Compatibility implementations remain registered for saved sources, not placement.
    static let replacedTypes: Set<String> = Set(["kwip.data.missing", "kwip.data.standardScalerFit", "kwip.data.imputerFit", "kwip.data.oneHotEncoderFit", "kwip.data.transform", "kwip.data.calculate", "kwip.data.regexExtract", "kwip.data.regexReplace", "kwip.data.split", "kwip.data.groupSplit", "kwip.data.temporalSplit", "kwip.data.selectRename", "kwip.data.cast", "kwip.data.filter", "kwip.data.filterExpression", "kwip.data.regexMatch", "dataset_source", "data_loader", "initialization", "gradient_policy", "trainability"]).union(RetiredTrainingRecipeMigration.componentIDs)
        .union(ActivationConfiguration.componentIDs).union(PoolingConfiguration.componentIDs).union(ConvolutionConfiguration.componentIDs).union(NormalizationConfiguration.componentIDs).union(TensorOperationPalette.replacedTypes).union(ReshapeAuthoringConfiguration.componentIDs).union(LinearConfiguration.componentIDs).union(RecurrentAuthoringConfiguration.componentIDs).union(OptimizerConfiguration.componentIDs).union(SchedulerConfiguration.componentIDs).union(LossConfiguration.componentIDs)
    private static func offered(_ entries: [ComponentRibbonEntry]) -> [ComponentRibbonEntry] {
        let offered=TensorOperationPalette.consolidated(entries:ReshapePalette.consolidated(entries:RecurrentCellPalette.consolidated(entries:RecurrentPalette.consolidated(entries:LinearPalette.consolidated(entries:NormalizationPalette.consolidated(entries:ConvolutionPalette.consolidated(entries:PoolingPalette.consolidated(entries:ActivationPalette.consolidated(entries:entries))))))))).filter { !replacedTypes.contains($0.typeIcon) }
        let priorities:[String:[String]] = [
            "Data":dataPriority,
            "Layers":["kwip.linear","bilinear","kwip.convolution","separable_conv2d","kwip.recurrent","kwip.recurrentCell","multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","attention_mask"],
            "Embeddings":["embedding","embedding_bag","bag_input"],
            "Conditioning":["kwip.normalization","dropout"],
            "Tensor layout":["kwip.reshape","kwip.tensor.axes","kwip.tensor.parts","unfold","fold"],
            "Tensor math":["matmul","einsum","gather"],
            "Merge":["concat","add","stack","parallel_features"],
            "Training":["workflow.train","workflow.saveModel","workflow.logisticRegression","workflow.kerasClassifier"],
            "Boundaries":["workflow.modelInput","workflow.modelDefinition","workflow.frozenModel"],
            "Evaluation":["workflow.evaluateTest","objective_check"]
        ]
        let values=offered.map {value in
            value.typeIcon == "dropout" ? ComponentRibbonEntry(id:value.id,title:value.title,shortTitle:value.shortTitle,purpose:value.purpose,category:"Conditioning",typeIcon:value.typeIcon):value
        }
        var queues:[String:[ComponentRibbonEntry]]=[:]
        for (category,members) in Dictionary(grouping:Array(values.enumerated()),by:{$0.element.category}) {
            let order=priorities[category] ?? []
            let rank=Dictionary(uniqueKeysWithValues:order.enumerated().map {($0.element,$0.offset)})
            queues[category]=members.sorted {
                let left=rank[$0.element.typeIcon] ?? order.count,right=rank[$1.element.typeIcon] ?? order.count
                return left == right ? $0.offset < $1.offset:left < right
            }.map(\.element)
        }
        var indices:[String:Int]=[:]
        return values.map {entry in
            let index=indices[entry.category,default:0];indices[entry.category]=index+1
            return queues[entry.category]![index]
        }
    }
    static func initialScope(restricted: Bool) -> Scope { restricted ? .all : .common }
    static func categories(_ entries:[ComponentRibbonEntry])->[String] {
        let available=offered(entries).reduce(into:[String]()) {if !$0.contains($1.category) {$0.append($1.category)}}
        let order=["Data","Boundaries","Layers","Embeddings","Activations","Pooling","Conditioning","Tensor layout","Tensor math","Merge","Training","Evaluation","Blocks"]
        return order.filter(available.contains)+available.filter {!order.contains($0)}
    }
    static func visible(_ entries:[ComponentRibbonEntry],category:String?,query:String)->[ComponentRibbonEntry] {
        visible(entries, scope: category.map(Scope.category) ?? .all, query: query)
    }
    static func visible(_ entries:[ComponentRibbonEntry],scope:Scope,query:String)->[ComponentRibbonEntry] {
        let entries = offered(entries)
        let query=query.trimmingCharacters(in:.whitespacesAndNewlines)
        if !query.isEmpty {
            return entries.filter { [$0.title,$0.shortTitle,$0.purpose,$0.category].contains { $0.localizedCaseInsensitiveContains(query) } }
        }
        switch scope {
        case .all: return entries
        case .category(let category):
            guard categories(entries).contains(category) else { return entries }
            return entries.filter { $0.category == category }
        case .common:
            let values = commonPriority.flatMap { type in entries.filter { $0.typeIcon == type } }
            // Specialized lessons and external catalogs must remain usable even
            // when none of the current curated operator types are present.
            return values.isEmpty ? entries : values
        }
    }
    static func groups(_ entries:[ComponentRibbonEntry],scope:Scope,query:String)->[Group] {
        let visible = visible(entries, scope: scope, query: query)
        return categories(entries).compactMap { category in
            let values = visible.filter { $0.category == category }
            return values.isEmpty ? nil : Group(category: category, entries: values)
        }
    }
}

/// One ribbon for every editor. Native previews carry registry IDs into the shared canvas.
struct ComponentRibbonView: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    let entries:[ComponentRibbonEntry]
    let isBusy:Bool
    let onAdd:(String)->Void
    var onDragStart:((String)->Void)? = nil
    @State private var scope:ComponentRibbonCatalog.Scope = .common
    @State private var query=""
    @State private var hovered:String?
    init(entries: [ComponentRibbonEntry], isBusy: Bool, onAdd: @escaping (String) -> Void,
         onDragStart: ((String) -> Void)? = nil, restricted: Bool = false) {
        self.entries = entries; self.isBusy = isBusy; self.onAdd = onAdd; self.onDragStart = onDragStart
        _scope = State(initialValue: ComponentRibbonCatalog.initialScope(restricted: restricted))
    }
    private var searching:Bool { !query.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty }
    var body: some View {
        VStack(spacing:0) {
            HStack(alignment:.center,spacing:12) {
                ThemedScrollView(.horizontal) {
                    HStack(spacing:4) {
                        categoryButton("Common",value:.common)
                        categoryButton("All",value:.all)
                        ForEach(ComponentRibbonCatalog.categories(entries),id:\.self) { categoryButton($0,value:.category($0)) }
                    }
                }.frame(height:28)
                TextField("Search all components",text:$query)
                    .font(.system(size:11)).flatField()
                    .frame(minWidth:130,maxWidth:220).frame(height:28).accessibilityLabel("Search all component categories")
            }.frame(height:32).padding(.horizontal,12)
            Rectangle().fill(MLTheme.muted.opacity(0.22)).frame(height:1)
            Color.clear.frame(height:5).accessibilityHidden(true)
            RibbonHorizontalScrollView {
                HStack(alignment:.top,spacing:8) {
                    let groups = ComponentRibbonCatalog.groups(entries,scope:scope,query:query)
                    ForEach(groups) { group in
                        let family = ComponentVisualFamily.resolve(type: group.entries[0].typeIcon)
                        VStack(alignment:.leading,spacing:5) {
                            Text(group.category).font(.system(size:9,weight:.semibold))
                                .foregroundStyle(family.primary).lineLimit(1).frame(height:11)
                            HStack(alignment:.top,spacing:10) { ForEach(group.entries) { tile($0) } }
                        }
                        .padding(8)
                        .background(family.secondary.opacity(0.08))
                        .overlay(Rectangle().stroke(family.primary.opacity(0.35),lineWidth:0.75))
                        .accessibilityElement(children:.contain)
                        .accessibilityLabel("\(group.category) component group")
                    }
                    if groups.isEmpty {
                        Text(entries.isEmpty ? "No components available in this editor." : "No components match this search.")
                            .font(.caption).foregroundStyle(MLTheme.muted).frame(height:96)
                        if searching { Button("Clear search") {query=""}.buttonStyle(.borderless).frame(height:96) }
                    }
                }.padding(.horizontal,12)
            }.frame(height:110)
            Rectangle().fill(MLTheme.muted.opacity(0.22)).frame(height:1)
        }.frame(height:149).appearanceChrome()
    }
    private func categoryButton(_ title:String,value:ComponentRibbonCatalog.Scope)->some View {
        let selected = searching ? value == .all : scope == value
        return Button {scope=value;query=""} label: {
            Text(title).font(.system(size:11,weight:selected ? .semibold : .regular))
                .foregroundStyle(selected ? Color.primary : MLTheme.muted)
                .padding(.horizontal,10).frame(height:26)
                .overlay(alignment:.bottom) { Rectangle().fill(selected ? appearanceTheme.accentColor : .clear).frame(height:2) }
                .overlay(alignment:.trailing) { Rectangle().fill(MLTheme.muted.opacity(0.16)).frame(width:0.5,height:14) }
        }.buttonStyle(.plain).accessibilityLabel("Show \(title) components").accessibilityValue(selected ? "Selected" : "")
            .help(value == .common ? "Curated common components available in this editor." : "Show \(title) components")
    }
    private func tile(_ entry:ComponentRibbonEntry)->some View {
        VStack(spacing:4) {
            ComponentIcon(type:entry.typeIcon,size:44)
                .opacity(hovered == entry.id ? 1 : 0.94)
                .contentShape(Rectangle())
                .onDrag {
                    guard !isBusy else { return NSItemProvider() }
                    onDragStart?(entry.id)
                    // Native String transfer is handled by the shared canvas drop destination.
                    // Its preview follows the pointer outside the ribbon and across windows.
                    return NSItemProvider(object: entry.id as NSString)
                } preview: {
                    ComponentIcon(type:entry.typeIcon,size:44)
                }
                .disabled(isBusy)
                .onHover {hovered=$0 ? entry.id : nil}
                .help(entry.title+" · "+entry.purpose)
                .accessibilityElement(children:.ignore)
                .accessibilityLabel(entry.title+" component. "+entry.purpose)
                .accessibilityAction(named:"Add to canvas") { if !isBusy {onAdd(entry.id)} }
            HStack(spacing:3) {
                Text(entry.shortTitle).font(.system(size:10,weight:.medium)).lineLimit(1).help(entry.title)
                Button {if !isBusy {onAdd(entry.id)}} label: {Image(systemName:"plus").font(.system(size:9,weight:.semibold)).frame(width:16,height:16)}
                    .buttonStyle(.plain).disabled(isBusy).accessibilityLabel("Add \(entry.title)")
            }.frame(height:16)
        }.frame(width:90)
    }
}

/// The scrollbar is a standalone native control: the application's quiet-scroll
/// probes can hide NSScrollView chrome without hiding this deliberate affordance.
private struct RibbonHorizontalScrollView<Content: View>: View {
    @StateObject private var scroll = RibbonScrollController()
    private let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        VStack(spacing:0) {
            ScrollView(.horizontal) {
                content.background(RibbonScrollProbe(controller:scroll).allowsHitTesting(false))
            }
            .scrollIndicators(.hidden)
            .frame(height:96)
            RibbonHorizontalScroller(controller:scroll)
                .frame(height:14).padding(.horizontal,12)
        }
    }
}

@MainActor private final class RibbonScrollController: ObservableObject {
    @Published private(set) var fraction=0.0
    @Published private(set) var proportion=1.0
    private weak var scrollView:NSScrollView?
    private var chrome:NativeScrollChromeController?
    private var observers:[NSObjectProtocol]=[]
    deinit { for observer in observers {NotificationCenter.default.removeObserver(observer)} }
    func attach(_ scrollView:NSScrollView?) {
        guard self.scrollView !== scrollView else { chrome?.refresh();refresh();return }
        for observer in observers {NotificationCenter.default.removeObserver(observer)}
        observers.removeAll();chrome=nil;self.scrollView=scrollView
        guard let scrollView else {refresh();return}
        // SwiftUI can restore its own scroller after .scrollIndicators(.hidden).
        // Scope the shared policy to this exact ribbon scroll view. The deliberate
        // standalone NSScroller is a sibling and remains visible and operable.
        chrome=NativeScrollChromeController(root:scrollView)
        let clip=scrollView.contentView
        clip.postsBoundsChangedNotifications=true
        if let document=scrollView.documentView {
            document.postsFrameChangedNotifications=true
            observers.append(NotificationCenter.default.addObserver(forName:NSView.frameDidChangeNotification,object:document,queue:.main) { [weak self] _ in
                DispatchQueue.main.async {self?.refresh()}
            })
        }
        observers.append(NotificationCenter.default.addObserver(forName:NSView.boundsDidChangeNotification,object:clip,queue:.main) { [weak self] _ in
            DispatchQueue.main.async {self?.refresh()}
        })
        refresh()
    }
    func refresh() {
        guard let scrollView,let document=scrollView.documentView else {set(fraction:0,proportion:1);return}
        let width=max(0,document.bounds.width),viewport=max(0,scrollView.contentView.bounds.width)
        let maximum=max(0,width-viewport)
        set(fraction:maximum>0 ? min(1,max(0,scrollView.contentView.bounds.minX/maximum)):0,
            proportion:width>0 ? min(1,viewport/width):1)
    }
    private func set(fraction:Double,proportion:Double) {
        if abs(self.fraction-fraction)>0.00001 {self.fraction=fraction}
        if abs(self.proportion-proportion)>0.00001 {self.proportion=proportion}
    }
    func scroll(_ sender:NSScroller) {
        guard let scrollView,let document=scrollView.documentView else {return}
        let clip=scrollView.contentView,maximum=max(0,document.bounds.width-clip.bounds.width)
        guard maximum>0 else {return}
        var x=clip.bounds.minX
        switch sender.hitPart {
        case .decrementLine:x-=40
        case .incrementLine:x+=40
        case .decrementPage:x-=clip.bounds.width*0.85
        case .incrementPage:x+=clip.bounds.width*0.85
        default:x=sender.doubleValue*maximum
        }
        clip.scroll(to:CGPoint(x:min(maximum,max(0,x)),y:clip.bounds.minY))
        scrollView.reflectScrolledClipView(clip);refresh()
    }
}

private struct RibbonScrollProbe:NSViewRepresentable {
    let controller:RibbonScrollController
    func makeNSView(context:Context)->RibbonScrollProbeView {RibbonScrollProbeView(controller:controller)}
    func updateNSView(_ view:RibbonScrollProbeView,context:Context) {view.capture()}
}
private final class RibbonScrollProbeView:NSView {
    let controller:RibbonScrollController
    private var scheduled=false
    init(controller:RibbonScrollController) {self.controller=controller;super.init(frame:.zero)}
    required init?(coder:NSCoder) {fatalError("init(coder:) has not been implemented")}
    override func hitTest(_ point:NSPoint)->NSView? {nil}
    override func viewDidMoveToSuperview() {super.viewDidMoveToSuperview();capture()}
    override func viewDidMoveToWindow() {super.viewDidMoveToWindow();capture()}
    override func layout() {super.layout();capture()}
    func capture() {
        guard !scheduled else {return};scheduled=true
        DispatchQueue.main.async { [weak self] in
            guard let self else {return};self.scheduled=false
            // Never traverse the full window and accidentally bind to the canvas
            // or category strip. This probe lives inside this ribbon's content.
            self.controller.attach(self.enclosingScrollView)
        }
    }
}
private struct RibbonHorizontalScroller:NSViewRepresentable {
    @ObservedObject var controller:RibbonScrollController
    func makeCoordinator()->Coordinator {Coordinator(controller:controller)}
    func makeNSView(context:Context)->NSScroller {
        let view=NSScroller(frame:CGRect(x:0,y:0,width:160,height:14))
        view.scrollerStyle = .legacy;view.controlSize = .small
        view.target=context.coordinator;view.action=#selector(Coordinator.scroll(_:))
        view.setAccessibilityLabel("Scroll component groups horizontally")
        return view
    }
    func updateNSView(_ view:NSScroller,context:Context) {
        view.doubleValue=controller.fraction;view.knobProportion=controller.proportion
        view.isEnabled=controller.proportion<1
    }
    @MainActor final class Coordinator:NSObject {
        let controller:RibbonScrollController
        init(controller:RibbonScrollController) {self.controller=controller}
        @objc func scroll(_ sender:NSScroller) {controller.scroll(sender)}
    }
}
