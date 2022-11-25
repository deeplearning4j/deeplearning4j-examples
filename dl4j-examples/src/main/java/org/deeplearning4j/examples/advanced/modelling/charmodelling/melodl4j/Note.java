/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.advanced.modelling.charmodelling.melodl4j;

import javax.sound.midi.*;

import com.google.common.base.Objects;

public class Note implements Comparable<Note> {
    private final int pitch;
    private final long startTick;
    private long endTick;
    private final Integer instrument;
    private Integer channel;
    private final int velocity;
    public Note(int pitch, long startTick, int instrument, int channel, int velocity) {
        this.pitch=pitch;
        this.startTick=startTick;
        this.instrument=instrument;
        this.channel=channel;
        this.velocity=velocity;
    }

    @Override
    public String toString() {
        return "Pitch " + pitch + " starting at " + startTick + " ending at " + endTick + " on channel "
	    + channel + " with instrument " + instrument + " (" + PlayMelodyStrings.programs[instrument] + ")"
	    + " and volume " + velocity;
    }
    public long getEndTick() {
        return endTick;
    }
    public void setEndTick(long endTick) {
        this.endTick = endTick;
    }
    public int getPitch() {
        return pitch;
    }
    public int getNoteValueInScale() {
        return pitch%12;
    }
    public long getStartTick() {
        return startTick;
    }
    public double getStartSeconds(double microsecondsPerTick) {
        return 1e-6*microsecondsPerTick*getStartTick();
    }
    public double getEndSeconds(double microsecondsPerTick) {
        return 1e-6*microsecondsPerTick*getEndTick();
    }
    public int getInstrument() {
        return instrument;
    }
    public int getChannel() {
        return channel;
    }
    public int getVelocity() {
        return velocity;
    }
    public long getDurationInTicks() {
        return endTick-startTick;
    }
    @Override
    public int compareTo(Note other) {
        long diffL=startTick-other.startTick;
        if (diffL!=0) {
            return diffL>0?1:-1;
        }
        int diff=pitch-other.pitch;
        if (diff!=0) {
            return diff;
        }
        diff = channel-other.channel;
        if (diff!=0) {
            return diff;
        }
        return 0;
    }
    @Override
    public boolean equals(Object obj) {
    	if (obj == null) {
    		return false;
    	} else if (! (obj instanceof Note)) {
    		return false;
    	}
        Note other=(Note)obj;
        return startTick==other.startTick  && pitch==other.pitch && Objects.equal(channel, other.channel);
    }

    @Override
    public int hashCode() {
    	return Objects.hashCode(startTick, pitch, channel);
    }

    public void addMidiEvents(Track track) throws InvalidMidiDataException {
        MidiMessage midiMessageStart=new ShortMessage(ShortMessage.NOTE_ON,channel,pitch,velocity);
        track.add(new MidiEvent(midiMessageStart,startTick));
        MidiMessage midiMessageEnd=new ShortMessage(ShortMessage.NOTE_OFF,channel,pitch,0);
        track.add(new MidiEvent(midiMessageEnd,endTick));
    }

    public double getDurationInSeconds(double microsecondsPerTick) {
        return 1e-6*microsecondsPerTick*(endTick-startTick);
    }

    public void setChannel(Integer channel) {
        this.channel=channel;
    }
} // class Note
