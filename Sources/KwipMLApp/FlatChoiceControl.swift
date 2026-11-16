import SwiftUI

struct FlatChoiceOption: Identifiable, Equatable {
    let id: String
    let title: String
    var isEnabled = true
}

enum FlatChoiceNavigation {
    enum Direction: CaseIterable, Equatable { case next, previous, first, last }
    static func initialID(options:[FlatChoiceOption],selection:String)->String? {
        options.first(where:{$0.isEnabled && $0.id == selection})?.id ?? options.first(where:{$0.isEnabled})?.id
    }
    static func move(options:[FlatChoiceOption],from current:String?,direction:Direction)->String? {
        let enabled=options.filter(\.isEnabled).map(\.id)
        guard !enabled.isEmpty else {return nil}
        switch direction {
        case .first:return enabled.first
        case .last:return enabled.last
        case .next:
            guard let current,let index=enabled.firstIndex(of:current) else {return enabled.first}
            return enabled[(index+1)%enabled.count]
        case .previous:
            guard let current,let index=enabled.firstIndex(of:current) else {return enabled.last}
            return enabled[(index+enabled.count-1)%enabled.count]
        }
    }
    static func commitID(options:[FlatChoiceOption],highlighted:String?,controlEnabled:Bool)->String? {
        guard controlEnabled,let highlighted else {return nil}
        return options.first(where:{$0.id == highlighted && $0.isEnabled})?.id
    }
}

/// Stable, themed choices without native menu tracking or a second system chevron.
struct FlatChoiceControl: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    let title: String
    let selection: String
    let options: [FlatChoiceOption]
    var placeholder = "Choose an option"
    var compact = false
    var accent: Color? = nil
    let onSelect: (String) -> Void
    @Environment(\.isEnabled) private var isEnabled
    @State private var expanded = false
    @State private var hoveredID: String?
    @State private var highlightedID: String?
    @State private var restoreTriggerOnDismiss = true
    @FocusState private var triggerFocused: Bool
    @FocusState private var listFocused: Bool
    private static let navigationKeys:Set<KeyEquivalent>=[.upArrow,.downArrow,.leftArrow,.rightArrow,.home,.end,.return,.space,.escape]

    private var controlAccent: Color { accent ?? appearanceTheme.accentColor }

    var selectedTitle: String {
        options.first(where: { $0.id == selection })?.title ?? placeholder
    }

    var body: some View {
        Button { if expanded { dismissChoices() } else { openChoices() } } label: {
            HStack(spacing: compact ? 4 : 8) {
                Text(selectedTitle).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                Image(systemName: expanded ? "chevron.up" : "chevron.down")
                    .font(.system(size: 9, weight: .semibold)).foregroundStyle(MLTheme.muted)
            }
            .font(.system(size: compact ? 11 : 12)).padding(compact ? 6 : 10)
            .background(MLTheme.background)
            .overlay(Rectangle().stroke(expanded || triggerFocused ? controlAccent : MLTheme.rule))
            .contentShape(Rectangle())
            .opacity(isEnabled ? 1 : 0.4)
        }
        .buttonStyle(.plain)
        // Edit interaction remains keyboard reachable when macOS button traversal is off.
        .focusable(isEnabled, interactions: .edit)
        .focused($triggerFocused)
        .focusEffectDisabled()
        .onKeyPress(keys:Self.navigationKeys) { handleKey($0) }
        .accessibilityLabel(title)
        .accessibilityValue(selectedTitle)
        .help(selectedTitle)
        .attachedDropdown(isPresented: $expanded, preferredHeight: min(300, CGFloat(max(1, options.count)) * 38), width: compact ? 160 : nil, onDismiss: { reason in
            if reason == .outsideInteraction { restoreTriggerOnDismiss = false }
        }) {
            VStack(alignment: .leading, spacing: 0) {
                ScrollViewReader { reader in
                ThemedScrollView {
                    VStack(spacing: 0) {
                        if options.isEmpty {
                            Text("No options available").font(.system(size: 12)).foregroundStyle(MLTheme.muted).padding(12)
                        }
                        ForEach(options) { option in
                            optionRow(option)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onChange(of:highlightedID) { _,id in if let id {reader.scrollTo(id,anchor:.center)} }
                .task {
                    await Task.yield()
                    guard expanded else {return}
                    if let highlightedID {reader.scrollTo(highlightedID,anchor:.center)}
                    listFocused=true
                }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity).background(MLTheme.panel)
            .overlay(Rectangle().stroke(MLTheme.rule))
            .focusable(isEnabled,interactions:.edit)
            .focused($listFocused)
            .focusEffectDisabled()
            .accessibilityElement(children:.contain)
            .accessibilityLabel("\(title) choices")
            .accessibilityValue(options.first(where:{$0.id == highlightedID})?.title ?? "No available choice")
            .onKeyPress(keys:Self.navigationKeys) { handleKey($0) }
            .onKeyPress(.tab) {
                // Leave Tab/Shift-Tab to normal traversal instead of trapping it
                // on the trigger after the popup is dismissed.
                restoreTriggerOnDismiss=false
                dismissChoices()
                return .ignored
            }
            .onExitCommand { dismissChoices() }
        }
        .onChange(of:expanded) { _,open in
            if !open {
                listFocused=false
                if restoreTriggerOnDismiss {triggerFocused=isEnabled}
            }
        }
        .onChange(of:options) { _,_ in
            if expanded,FlatChoiceNavigation.commitID(options:options,highlighted:highlightedID,controlEnabled:isEnabled) == nil {
                highlightedID=FlatChoiceNavigation.initialID(options:options,selection:selection)
            }
        }
        .onChange(of: isEnabled) { _, enabled in if !enabled { expanded = false;triggerFocused=false } }
    }
    private func optionRow(_ option:FlatChoiceOption)->some View {
        Button {choose(option.id)} label: {optionLabel(option)}
            .buttonStyle(.plain).disabled(!option.isEnabled)
            // The list owns keyboard focus; rows retain pointer/AX activation.
            .focusable(false)
            .id(option.id)
            .onHover {hoveredID = $0 ? option.id : nil}
            .accessibilityLabel(option.title)
            .accessibilityValue(option.id == selection ? "Selected" : "")
    }
    private func optionLabel(_ option:FlatChoiceOption)->some View {
        let highlighted=highlightedID == option.id
        let fill:Color=(hoveredID == option.id || highlighted) ? controlAccent.opacity(0.12) : .clear
        let border:Color=highlighted ? controlAccent : .clear
        return HStack(spacing:8) {
            Image(systemName:"checkmark").font(.system(size:10,weight:.semibold))
                .foregroundStyle(controlAccent).opacity(option.id == selection ? 1 : 0)
                .frame(width:13)
            Text(option.title).font(.system(size:12)).multilineTextAlignment(.leading)
            Spacer(minLength:0)
        }
        .padding(.horizontal,12).padding(.vertical,10)
        .frame(maxWidth:.infinity,alignment:.leading)
        .background(fill)
        .overlay(Rectangle().stroke(border,lineWidth:1))
        .contentShape(Rectangle()).opacity(option.isEnabled ? 1 : 0.4)
    }
    private func openChoices() {
        guard isEnabled else {return}
        restoreTriggerOnDismiss=true
        highlightedID=FlatChoiceNavigation.initialID(options:options,selection:selection)
        // Opening must not depend on a FocusState callback: nested inspector
        // focus scopes may decline the trigger request. The attached menu establishes
        // its own list focus after presentation.
        triggerFocused=true
        expanded=true
    }
    private func dismissChoices() {expanded=false}
    private func choose(_ id:String?) {
        guard let selected=FlatChoiceNavigation.commitID(options:options,highlighted:id,controlEnabled:isEnabled) else {return}
        dismissChoices();onSelect(selected)
    }
    private func handleKey(_ press:KeyPress)->KeyPress.Result {
        guard isEnabled,press.modifiers.intersection([.command,.control,.option]).isEmpty else {return .ignored}
        if press.key == .escape {
            guard expanded else {return .ignored}
            dismissChoices();return .handled
        }
        if press.key == .return || press.key == .space {
            if press.phase == .repeat {return .handled}
            if expanded {choose(highlightedID)} else {openChoices()}
            return .handled
        }
        let direction:FlatChoiceNavigation.Direction
        switch press.key {
        case .upArrow,.leftArrow:direction = .previous
        case .downArrow,.rightArrow:direction = .next
        case .home:direction = .first
        case .end:direction = .last
        default:return .ignored
        }
        if !expanded {
            openChoices()
            if direction == .next || direction == .previous {return .handled}
        }
        highlightedID=FlatChoiceNavigation.move(options:options,from:highlightedID,direction:direction)
        return .handled
    }
}
