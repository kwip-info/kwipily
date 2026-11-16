import SwiftUI

struct FormulaTemporalInformationView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Typed date and time", systemImage: "calendar").fontWeight(.semibold)
            Text("Parse creates Date, DateTime or Time values. Date is a calendar day; Time is a clock reading without a date or timezone. DateTime identifies an instant. Convert timezone changes how that instant is displayed, without moving it.")
            Label("Explicit patterns and timezones", systemImage: "clock").fontWeight(.semibold)
            Text("ISO uses YYYY-MM-DD, HH:MM:SS and YYYY-MM-DDTHH:MM:SS with an optional offset. Numeric custom patterns use %Y year, %m month, %d day, %H hour, %M minute, %S second, %f microseconds and %z offset. Locale is invariant; month names, natural language and calendar arithmetic are unavailable. Naive DateTime values require a source IANA timezone, such as America/Chicago. Machine timezone is never inferred.")
            Text("At a daylight-saving clock change, a local time may identify two instants or no instant. Choose an explicit ambiguous/nonexistent policy. Earliest and Latest select by instant, not by display order. Missing policies apply only to their named condition; they do not hide precision loss.")
            Label("Precision and useful features", systemImage: "number").fontWeight(.semibold)
            Text("DateTime and Time retain six fractional digits. Extra precision is refused without rounding. Difference subtracts the second column from the first in elapsed units; Date uses whole calendar days. Exact integer refuses fractional results. Decimal explicitly uses float64. Extract creates numeric features such as hour or weekday (Monday = 1); model mapping never treats stored timestamps as implicit numeric features.")
        }
    }
}
