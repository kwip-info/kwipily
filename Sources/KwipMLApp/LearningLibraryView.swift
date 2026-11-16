import SwiftUI
import KwipMLCore

/// Application-level learning dashboard. Progress is a projection of local evidence,
/// never an inferred mastery score or a count of opened examples.
struct LearningLibraryView: View {
    @Environment(\.colorScheme) private var colorScheme
    @ObservedObject var store: WorkspaceStore
    @ObservedObject var affineLearning: AffineLearningStore
    @ObservedObject var normalizationLearning: NormalizationLearningStore
    @ObservedObject var tensorLearning: TensorLearningStore
    @ObservedObject var navigation: LearningLibraryNavigation
    private var section: String { navigation.section }
    private var search: String { navigation.search }
    @State private var resetRevision = UUID()
    @State private var records: [LearningOverviewRecord] = []
    private let subjects = LearningLibraryPresentation.topics
    private var refreshIdentity: String { resetRevision.uuidString + store.lessonProjects.map { $0.id + String($0.updatedAt.timeIntervalSince1970) }.joined(separator: ":") }
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Label("Learn & explore", systemImage: "book.closed").font(.system(size: 16, weight: .semibold))
                Spacer()
                TextField("Search lessons, concepts and components", text: $navigation.search).textFieldStyle(.plain)
                    .padding(8).frame(maxWidth: 440).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    .accessibilityLabel("Search learning library")
                if store.activeProjectID != nil {
                    Button("Return to canvas") { store.focusOwningProject?() }
                }
                Button { ApplicationNavigationWindows.shared.showSettings() } label: { Image(systemName: "gearshape") }
                    .accessibilityLabel("Global settings")
            }.padding(14)
            Divider()
            HStack(spacing: 0) {
                sidebar.frame(width: 215)
                Divider()
                ScrollViewReader { scroll in
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if let entryID = navigation.directEntryID {
                            VStack(alignment: .leading, spacing: 18) {
                                Button { navigation.browseLibrary() } label: { Label("Back to " + section.lowercased(), systemImage: "chevron.left") }
                                Button("Open on canvas") {openOnCanvas(entryID)}
                                Text(LearningLibraryPresentation.items.first(where: {$0.id==entryID})?.title ?? "Lesson").font(.title2)
                            }.accessibilityIdentifier("learning.direct." + entryID)
                        } else if section == "Overview" && search.isEmpty { overview }
                        else if section == "Saved lessons" {
                            savedLessons
                        }
                        else {
                            Text(search.isEmpty ? section : "Search results").font(.system(size: 24, weight: .semibold))
                            discoveryFilters
                            LearnCatalogContent(store: store, affineLearning: affineLearning, normalizationLearning: normalizationLearning,
                                tensorLearning: tensorLearning, section: section == "Overview" ? "Library" : section, search: search,
                                typeFilter: navigation.typeFilter, frameworkFilter: navigation.frameworkFilter,
                                openEntry: { openOnCanvas($0) })
                        }
                    }.padding(22).frame(maxWidth: .infinity, alignment: .leading)
                }
                .id(section + ":" + (navigation.directEntryID ?? ""))
                .onChange(of: navigation.directEntryID) { _, entry in
                    if entry == nil, let anchor = navigation.browseAnchorID {
                        Task { @MainActor in await Task.yield(); scroll.scrollTo(anchor, anchor: .center) }
                    }
                }
                }
            }
            Divider()
            HStack {
                Text("Progress saved on this Mac")
                Spacer()
                Text("Verified = assessment passed · Practice checked ≠ assessment completion")
            }.font(.system(size: 11)).foregroundStyle(MLTheme.muted).padding(10)
        }
        .background(MLTheme.background).buttonStyle(FlatButtonStyle())
        .onChange(of: navigation.search) { _, value in if !value.isEmpty { navigation.browseLibrary() } }
        .onReceive(NotificationCenter.default.publisher(for: .learningProgressDidReset)) { event in
            guard event.object as? String == store.workspaceDirectory.standardizedFileURL.path else { return }
            resetRevision = UUID()
        }
        .onReceive(store.lessonReading.$revision) {_ in resetRevision=UUID()}
        .task(id: refreshIdentity) {
            let projects = store.lessonProjects.filter {!store.lessonReading.isUnsaved(projectID:$0.id)}
            let directory = store.workspaceDirectory.appendingPathComponent("learning/projects")
            let savedIDs=Set(projects.filter {project in store.lessonReading.entry(projectID:project.id).map {store.lessonReading.record($0).savedProjectID==project.id} == true}.map(\.id))
            let result = await Task.detached(priority: .utility) { LearningOverviewRecord.read(projects: projects, directory: directory,savedLessonIDs:savedIDs) }.value
            guard !Task.isCancelled else { return }; records = result
        }
    }
    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 4) {
            navigation("Overview", symbol: "house")
            navigation("Library", symbol: "books.vertical")
            navigation("Saved lessons", symbol: "bookmark")
            Divider().padding(.vertical, 14)
            Text("TOPICS").font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted).padding(8)
            ForEach(subjects, id: \.self) { topic in
                navigation(topic, symbol: LearningLibraryPresentation.symbol(topic), color: colorScheme == .dark ? LearningLibraryPresentation.family(topic).primary : LearningLibraryPresentation.family(topic).secondary)
            }
            Spacer()
        }.padding(12).background(MLTheme.panel)
    }
    private func navigation(_ target: String, title: String? = nil, symbol: String, color: Color = MLTheme.rose) -> some View {
        Button { navigation.showSection(target) } label: {
            HStack(spacing: 10) {
                Image(systemName: symbol).foregroundStyle(target == "Overview" || target == "Learning paths" || target == "Library" || target == "Saved lessons" ? color : .white)
                    .frame(width: 22, height: 24).background(subjects.contains(target) ? color : Color.clear)
                Text(title ?? target).font(.system(size: 12)); Spacer(minLength: 0)
            }.padding(.vertical, 9).padding(.horizontal, 8).frame(maxWidth: .infinity, alignment: .leading)
                .background(section == target ? color.opacity(0.12) : Color.clear)
        }.buttonStyle(.plain).accessibilityLabel("Learn " + (title ?? target))
    }
    private var discoveryFilters: some View {
        HStack(spacing: 10) {
            FlatChoiceControl(title: "Learning material type", selection: navigation.typeFilter,
                options: (["All types"] + LearningLibraryPresentation.types).map { .init(id: $0, title: $0) }) { navigation.typeFilter = $0 }
            FlatChoiceControl(title: "Learning framework", selection: navigation.frameworkFilter,
                options: (["All frameworks"] + LearningLibraryPresentation.frameworks).map { .init(id: $0, title: $0) }) { navigation.frameworkFilter = $0 }
            Text("\(LearningLibraryPresentation.matching(section: section == "Overview" ? "Library" : section, search: search, type: navigation.typeFilter, framework: navigation.frameworkFilter).count) entries")
                .font(.caption).foregroundStyle(MLTheme.muted).fixedSize()
        }
    }
    private func box<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.system(size: 15, weight: .semibold))
            content()
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
            .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private var overview: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Learn by building").font(.system(size: 25, weight: .semibold))
                    Text("Choose a starting point, then explore it on the canvas.").font(.callout).foregroundStyle(MLTheme.muted)
                }
                Spacer(); Button("Browse library") { navigation.showSection("Library") }
            }
            box("Continue learning") {
                if let latest = records.first {
                    HStack(spacing: 12) {
                        Image(systemName: "square.grid.2x2").foregroundStyle(MLTheme.rose)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(latest.name).fontWeight(.semibold)
                            Text(latest.status.rawValue + " · " + latest.updatedAt.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(MLTheme.muted)
                        }
                        Spacer(); Button("Resume") { resume(latest.id) }.disabled(store.isBusy)
                    }
                } else {
                    HStack { Text("No saved lesson yet. Choose a starting point below.").font(.caption).foregroundStyle(MLTheme.muted); Spacer(); Button("Explore foundations") { navigation.showSection("Foundations") } }
                }
            }
            box("Explore next") {
                VStack(spacing: 0) {
                    ForEach(LearningLibraryPresentation.suggestions, id: \.journey.id) { suggestion in
                        if let item = LearningLibraryPresentation.items.first(where: { $0.id == suggestion.step.entryID }) {
                            LearningDiscoveryRow(item: item, reading:store.lessonReading, purpose: suggestion.step.advice) { openOnCanvas(item.id) }
                        }
                    }
                }
            }.help("Authored starting points, independent of assessment progress.")
            HStack {
                Text("\(records.count) saved lessons · \(records.filter { $0.status == .verified }.count) verified · \(records.filter { $0.status == .needsRecheck }.count) need recheck")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                Spacer(); Button("Saved lessons") { navigation.showSection("Saved lessons") }
            }
            Text("Reading progress is separate from checked construction results.")
                .font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    private var savedLessons: some View {
        let terms = search.split(whereSeparator: \.isWhitespace).map(String.init)
        let matchingRecords = records.filter { record in
            let text = record.name + " " + record.status.rawValue
            return terms.allSatisfy { text.localizedCaseInsensitiveContains($0) }
        }
        return box("Saved lessons") {
            if records.isEmpty { Text("Your saved lessons will appear here.").foregroundStyle(MLTheme.muted) }
            else if matchingRecords.isEmpty { Text("No saved lessons match your search.").foregroundStyle(MLTheme.muted) }
            ForEach(matchingRecords) { record in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(record.name).fontWeight(.medium)
                        Text(record.status.rawValue).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Resume") { resume(record.id) }.disabled(store.isBusy)
                    Button("Unsave") {
                        if store.lessonReading.unsave(projectID:record.id) {records.removeAll {$0.id==record.id}}
                        else {store.statusMessage=store.lessonReading.error ?? "Bookmark could not be removed."}
                    }.help("Remove bookmark; keep the canvas and assessment history")
                }
                Divider()
            }
        }
    }
    private func openOnCanvas(_ id:String) {ApplicationNavigationWindows.shared.openLessonOnCanvas(entryID:id)}
    private func resume(_ id: String) {
        guard let record = store.lessonProjects.first(where: { $0.id == id }) else { return }
        store.openManagedProject(record)
    }
}
