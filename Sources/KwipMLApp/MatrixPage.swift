import KwipMLCore

/// Bounded row-major projection; changing shapes safely clamps stale page selections.
struct MatrixPage {
    let tensor: LearningTensor
    let rows: Range<Int>
    let columns: Range<Int>
    let rowPageCount: Int
    let columnPageCount: Int
    init?(tensor: LearningTensor, rowPage: Int, columnPage: Int, pageSize: Int = 4) {
        guard tensor.wellFormed, tensor.shape.count == 2, (1...4).contains(pageSize) else { return nil }
        self.tensor = tensor
        let height = tensor.shape[0], width = tensor.shape[1]
        rowPageCount = (height + pageSize - 1) / pageSize
        columnPageCount = (width + pageSize - 1) / pageSize
        let row = min(max(0, rowPage), rowPageCount - 1) * pageSize
        let column = min(max(0, columnPage), columnPageCount - 1) * pageSize
        rows = row..<min(height, row + pageSize)
        columns = column..<min(width, column + pageSize)
    }
    func value(row: Int, column: Int) -> Double? {
        guard row >= 0, row < tensor.shape[0], column >= 0, column < tensor.shape[1] else { return nil }
        return tensor.values[row * tensor.shape[1] + column]
    }
}
