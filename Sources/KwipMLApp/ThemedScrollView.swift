import AppKit
import SwiftUI

/// A native scrolling surface with quiet chrome, including when macOS is set to
/// always display scrollbars. Wheel, trackpad and accessibility scrolling remain native.
struct ThemedScrollView<Content: View>: View {
    private let axes: Axis.Set
    private let content: Content

    init(_ axes: Axis.Set = .vertical, @ViewBuilder content: () -> Content) {
        self.axes = axes
        self.content = content()
    }

    var body: some View {
        ScrollView(axes) {
            content.background(NativeScrollChromeProbe().allowsHitTesting(false))
        }
        .scrollIndicators(.hidden)
    }
}

@MainActor enum NativeScrollChrome {
    static func hideSystemScrollers(in scroll: NSScrollView) {
        guard !(scroll is CapturedTableScrollView) else { return }
        if scroll.scrollerStyle != .overlay { scroll.scrollerStyle = .overlay }
        if scroll.hasHorizontalScroller { scroll.hasHorizontalScroller = false }
        if scroll.hasVerticalScroller { scroll.hasVerticalScroller = false }
        scroll.autohidesScrollers = true
        scroll.horizontalScroller?.isHidden = true
        scroll.verticalScroller?.isHidden = true
    }
}

private struct NativeScrollChromeProbe: NSViewRepresentable {
    func makeNSView(context: Context) -> NativeScrollChromeView { NativeScrollChromeView() }
    func updateNSView(_ view: NativeScrollChromeView, context: Context) { view.applyChrome() }
}

private final class NativeScrollChromeView: NSView {
    private weak var observedRoot:NSView?
    private var controller:NativeScrollChromeController?
    private var updateScheduled=false
    override func hitTest(_ point:NSPoint)->NSView? {nil}
    override func viewDidMoveToWindow() {super.viewDidMoveToWindow();applyChrome()}
    override func viewDidMoveToSuperview() {super.viewDidMoveToSuperview();applyChrome()}
    override func layout() {super.layout();applyChrome()}
    func applyChrome() {
        guard !updateScheduled else{return};updateScheduled=true
        DispatchQueue.main.async { [weak self] in
            guard let self else{return};self.updateScheduled=false
            // SwiftUI may mount a background probe beside its native scrolling host.
            // Prefer the enclosing scroll; otherwise inspect only this app window.
            guard let root=self.enclosingScrollView ?? self.window?.contentView else{return}
            if self.observedRoot !== root {self.observedRoot=root;self.controller=NativeScrollChromeController(root:root)}
            self.controller?.refresh()
        }
    }
}

/// Watches native scroller flags because SwiftUI may reapply them after mounting.
/// This never changes system preferences or affects views outside the supplied root.
@MainActor final class NativeScrollChromeController {
    private weak var root:NSView?
    private var observations:[ObjectIdentifier:[NSKeyValueObservation]]=[:]
    private var scheduled=false
    init(root:NSView) {self.root=root;refresh()}
    func refresh() {
        guard let root else {observations.removeAll();return}
        var scrolls:[NSScrollView]=[]
        func visit(_ view:NSView) {
            if let scroll=view as? NSScrollView, !(scroll is CapturedTableScrollView) {scrolls.append(scroll)}
            for child in view.subviews {visit(child)}
        }
        visit(root)
        let identities=Set(scrolls.map {ObjectIdentifier($0)})
        observations=observations.filter {identities.contains($0.key)}
        for scroll in scrolls {
            let identity=ObjectIdentifier(scroll)
            if observations[identity] == nil {
                observations[identity]=[
                    scroll.observe(\.hasVerticalScroller,options:[.new]) { [weak self] _,_ in DispatchQueue.main.async { self?.scheduleRefresh() } },
                    scroll.observe(\.hasHorizontalScroller,options:[.new]) { [weak self] _,_ in DispatchQueue.main.async { self?.scheduleRefresh() } },
                    scroll.observe(\.scrollerStyle,options:[.new]) { [weak self] _,_ in DispatchQueue.main.async { self?.scheduleRefresh() } }
                ]
            }
            NativeScrollChrome.hideSystemScrollers(in:scroll)
        }
    }
    private func scheduleRefresh() {
        guard !scheduled else{return};scheduled=true
        DispatchQueue.main.async { [weak self] in self?.scheduled=false;self?.refresh() }
    }
}
