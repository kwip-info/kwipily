import Foundation

/// Query presentation lives beside the retained page, so remounting its SwiftUI
/// view cannot relabel filtered/sorted rows as the default unfiltered table.
struct TableViewNavigationState: Equatable {
    private(set) var offset = 0
    private(set) var search = ""
    private(set) var sort = TableGridSort()
    private(set) var previousOffsets: [Int] = []

    mutating func begin(offset requested: Int, search text: String, sort ordering: TableGridSort,
                        pageOffset: Int?, nextOffset: Int?) {
        if text != search || ordering != sort || requested == 0 {
            previousOffsets = []
        } else if requested == previousOffsets.last {
            previousOffsets.removeLast()
        } else if requested != offset {
            if let pageOffset, nextOffset == requested {
                previousOffsets.append(pageOffset)
            } else {
                // A direct jump has no observed predecessor. Never invent pages.
                previousOffsets = []
            }
        }
        offset = requested; search = text; sort = ordering
    }
}
