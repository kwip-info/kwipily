import SwiftUI

/// Discovery metadata only. A row never opens a project or asserts completion.
@MainActor enum LearningLibraryPresentation {
    static let topics = ["Foundations", "Data preparation", "Tensors & linear algebra", "Layers & activations", "Training & evaluation"]
    private static let componentEntries = LearnCatalogContent.displayedComponentExampleEntries
    static let items = LearnCatalogContent.items + componentEntries.map(\.item)
    private static let componentPurposes = Dictionary(uniqueKeysWithValues: LearnCatalogContent.componentExampleEntries.map { ($0.componentID, $0.learning.subtitle) })
    private static let practicePurposes: [String: String] = [
        "embedding-bag": "Combine grouped category IDs into one learned vector per observation.",
        "program-export": "Save and reload a computation with declared input size rules.",
        "spatial-patches": "Extract local windows and inspect how overlapping regions combine.",
        "tensor-routing": "Select positions, split arrays and compare named arrangements.",
        "tensor-composition": "Combine arrays and reorder their directions without losing observation identity.",
        "adaptive-pooling": "Choose an output size and inspect the input regions it summarizes.",
        "transposed-convolution": "Compare how shared windows scatter and overlap in the output.",
        "gradient-examples": "Inspect worked comparisons of frozen settings and measured influence.",
        "initialization-examples": "Compare reproducible starting values before training changes them.",
        "checkpoint-examples": "Inspect restoring weights, fresh fine-tuning and exact continuation.",
        "sklearn": "Prepare familiar observations and train a local scikit-learn classifier.",
        "tensorflow": "Build and run a scoped TensorFlow classifier on the local runtime.",
        "join": "Match table rows with explicit keys and inspect unmatched observations.",
        "aggregate": "Group observations and inspect calculated summaries.",
        "separable": "Compare depthwise windows with pointwise channel mixing.",
        "depthwise": "Inspect independent channel windows and their output multiplier.",
        "transpose1d": "Follow overlapping windows into a one-dimensional output.",
        "transpose2d": "Follow overlapping windows into a spatial output.",
        "transpose3d": "Follow overlapping windows into a volume output."
    ]
    static let types = Array(Set(items.map(\.kind))).sorted()
    static let frameworks = Array(Set(items.map(\.backend))).sorted()
    static func matching(section: String, search: String, type: String = "All types", framework: String = "All frameworks") -> [LearnCatalogItem] {
        items.filter { $0.matches(section: section, search: search) && (type == "All types" || $0.kind == type) && (framework == "All frameworks" || $0.backend == framework) }
    }
    static func purpose(_ item: LearnCatalogItem) -> String {
        if let step = LearningJourney.all.flatMap({ $0.steps + $0.optionalSteps }).first(where: { $0.entryID == item.id }) { return step.advice }
        if let purpose = practicePurposes[item.id] ?? componentPurposes[item.id] { return purpose }
        if item.id == "train-data-access" { return TrainDataAccessLessonContent.objective }
        if item.id == "preprocess" {return PreprocessLessonContent.objective}
        if item.id == "formula" {return FormulaLessonContent.objective}
        if item.id == "partition" { return PartitionLessonContent.objective }
        if item.id == "filter" { return FilterLessonContent.objective }
        if item.id == "columns" { return ColumnsLessonContent.objective }
        if item.id == "document-classification" {return "Fit a private document classifier, inspect held-out mistakes, save its text preparation and predict unseen documents offline."}
        if item.id == "extract-text" {return "Read printed English from scanned images, inspect confidence and source regions, then reuse document text in local workflows."}
        if item.id == "prepare-media" {return "Prepare image pixels and PCM audio windows, then compare captured sources, tensor shapes and padding masks in View."}
        if item.id == "file-input" { return FileInputLessonContent.objective }
        return "Explore \(item.title.lowercased()) through this \(item.kind.lowercased())."
    }
    static func family(_ topic: String) -> ComponentVisualFamily {
        switch topic {
        case "Data preparation": return .data
        case "Tensors & linear algebra": return .tensorLayout
        case "Layers & activations": return .layers
        case "Training & evaluation": return .training
        default: return .blocks
        }
    }
    static func symbol(_ topic: String) -> String {
        switch topic {
        case "Data preparation": return "tablecells"
        case "Tensors & linear algebra": return "square.grid.3x3"
        case "Layers & activations": return "square.stack.3d.up"
        case "Training & evaluation": return "chart.xyaxis.line"
        default: return "lightbulb"
        }
    }
    static func firstItem(in journey: LearningJourney) -> LearnCatalogItem? {
        guard let id = journey.steps.first?.entryID else { return nil }
        return items.first { $0.id == id }
    }
    static let suggestions = Array(LearningJourney.all.prefix(3)).compactMap { journey -> (journey: LearningJourney, step: LearningJourney.Step)? in
        guard let step = journey.steps.first else { return nil }; return (journey, step)
    }
}

struct LearningDiscoveryRow: View {
    let item: LearnCatalogItem
    var reading: LessonReadingStore? = nil
    var purpose: String? = nil
    let open: () -> Void
    @State private var assessmentsExpanded=false
    @Environment(\.colorScheme) private var colorScheme
    private var color: Color { let family = LearningLibraryPresentation.family(item.category); return colorScheme == .dark ? family.primary : family.secondary }
    private var assessmentChoices:[ConstructionLessonCatalog.Choice] {
        let choices=ConstructionLessonCatalog.groups[item.id] ?? [];return choices.count>1 ? choices:[]
    }
    var body: some View {
        VStack(alignment:.leading,spacing:0) {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: LearningLibraryPresentation.symbol(item.category)).font(.system(size: 18)).foregroundStyle(Color.white)
                .frame(width: 36, height: 40).background(color)
            VStack(alignment: .leading, spacing: 4) {
                Text(item.title).font(.system(size: 13, weight: .semibold)).lineLimit(2)
                Text(purpose ?? LearningLibraryPresentation.purpose(item)).font(.system(size: 12)).foregroundStyle(MLTheme.muted).lineLimit(1)
                    .help(purpose ?? LearningLibraryPresentation.purpose(item))
                Text(item.backend).font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }.frame(maxWidth: .infinity, alignment: .leading)
            if let reading {LessonReadingBadge(entryID:item.id,reading:reading)}
            Text(item.kind).font(.system(size: 10, weight: .medium)).foregroundStyle(color)
                .padding(.horizontal, 7).padding(.vertical, 4).background(color.opacity(0.10))
            if assessmentChoices.isEmpty {
                Button("Open on canvas", action: open).accessibilityLabel("Open " + item.title).accessibilityIdentifier("learning.open." + item.id)
            } else {
                Button {assessmentsExpanded.toggle()} label: {
                    Label("Assessments",systemImage:assessmentsExpanded ? "chevron.up":"chevron.down")
                }.accessibilityLabel((assessmentsExpanded ? "Collapse":"Expand") + " assessments for " + item.title)
                    .accessibilityValue(assessmentsExpanded ? "Expanded":"Collapsed")
            }
        }.padding(.vertical, 8).padding(.horizontal, 12)
        if assessmentsExpanded,!assessmentChoices.isEmpty {
            VStack(alignment:.leading,spacing:0) {
                ForEach(assessmentChoices,id:\.id) {choice in
                    HStack {Text(choice.title).font(.system(size:12));Spacer();Button("Open assessment") {
                        ApplicationNavigationWindows.shared.openLessonOnCanvas(entryID:item.id,operation:choice.id)
                    }.accessibilityLabel("Open assessment: " + choice.title)}.padding(.vertical,6)
                }
            }.font(.caption).padding(.leading,60).padding(.trailing,12).padding(.bottom,12)
        }
        }.background(MLTheme.background)
            .overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
            .accessibilityIdentifier("learning.catalog." + item.id).id(item.id)
    }
}

/// A path is an authored reading order, never a progress or mastery claim.
struct LearningPathRow: View {
    let journey: LearningJourney
    var selected = false
    let open: () -> Void
    @Environment(\.colorScheme) private var colorScheme
    private var topic: String { LearningLibraryPresentation.firstItem(in: journey)?.category ?? "Foundations" }
    private var color: Color {
        let family = LearningLibraryPresentation.family(topic)
        return colorScheme == .dark ? family.primary : family.secondary
    }
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: LearningLibraryPresentation.symbol(topic)).font(.system(size: 18)).foregroundStyle(Color.white)
                .frame(width: 36, height: 40).background(color)
            VStack(alignment: .leading, spacing: 4) {
                Text(journey.title).font(.system(size: 13, weight: .semibold)).lineLimit(2)
                Text(journey.objective).font(.system(size: 12)).foregroundStyle(MLTheme.muted).lineLimit(1).help(journey.objective)
            }.frame(maxWidth: .infinity, alignment: .leading)
            Text("\(journey.steps.filter { !$0.isOptional }.count) stops").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            Button("View path", action: open).accessibilityLabel("View " + journey.title + " path")
        }.padding(.vertical, 8).padding(.horizontal, 12)
            .background(selected ? color.opacity(0.08) : MLTheme.background)
            .overlay(alignment: .bottom) { Rectangle().fill(selected ? color : MLTheme.rule).frame(height: 1) }
            .accessibilityIdentifier("learning.journey." + journey.id)
    }
}
